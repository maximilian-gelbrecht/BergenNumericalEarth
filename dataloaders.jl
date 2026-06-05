# Point-wise / column-wise dataloaders for training Lux.jl models on global fields.
#
# Idea
# ----
# We have several global NetCDF input fields (predictors) and one global NetCDF
# target field. We do NOT train a global model; instead every single grid point
# (optionally at every time step) is an independent training sample. The model
# maps the input values at one grid point to the target at the *same* grid point.
#
# Assumed in-memory layout after loading (this is what NCDatasets gives you when
# the file is written lat × lon × [level] × time, see `load_variable`):
#
#     3D variable:  (lat, lon, time)              -> contributes 1   feature
#     4D variable:  (lat, lon, level, time)       -> contributes Nlev features
#                                                     (or 1 if a single `level` is selected)
#
# All fields are flattened to a `(features, samples)` matrix. The sample axis is
# the LAST axis, which is exactly what Lux models and `MLUtils.DataLoader` expect
# (Lux batches over the trailing dimension). The flattening order is identical for
# every field, so inputs and target stay aligned point-for-point.

using NCDatasets, MLUtils, Random

# ----------------------------------------------------------------------------
# Variable specification
# ----------------------------------------------------------------------------

"""
    NetCDFVar(; path, name, level=nothing)

Points at one variable `name` inside the NetCDF file at `path`.

`level` selects along the vertical dimension of a 4D (lat, lon, level, time) field:
  * `nothing`  – keep all levels (each level becomes a separate feature),
                 or no-op for a 3D field that has no level dimension,
  * `Integer`  – take a single level, e.g. a surface or bottom-soil index,
  * a range / vector of indices – keep a subset of levels.
"""
Base.@kwdef struct NetCDFVar
    path::String
    name::String
    level::Any = nothing
end

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

"""
    load_variable(path, name; level=nothing) -> Array{Float32}

Read a variable from a NetCDF file as a dense `Float32` array. `_FillValue` /
`missing` entries are converted to `NaN` so they can be filtered later. If `level`
is given and the field is 4D, that level (or range of levels) is selected.
"""
function load_variable(path::AbstractString, name::AbstractString; level = nothing)
    arr = NCDataset(path) do ds
        Array(ds[name])                      # (lat, lon, [level], time)
    end
    arr = to_float32_nan(arr)

    if level !== nothing && ndims(arr) == 4
        arr = arr[:, :, level, :]            # Integer level -> drops the level dim
    end
    return arr
end

load_field(v::NetCDFVar) = load_variable(v.path, v.name; level = v.level)

"Convert any array (possibly holding `missing`) to a dense `Float32` array with `NaN` for missings."
function to_float32_nan(arr::AbstractArray)
    out = Array{Float32}(undef, size(arr))
    @inbounds for i in eachindex(arr, out)
        x = arr[i]
        out[i] = ismissing(x) ? NaN32 : Float32(x)
    end
    return out
end

# ----------------------------------------------------------------------------
# Flattening to (features, samples)
# ----------------------------------------------------------------------------

"""
    flatten_pointwise(arr) -> Matrix (features, samples)

Flatten a global field so each column is one grid point at one time step.

  * 3D `(lat, lon, time)`         -> `(1, lat*lon*time)`
  * 4D `(lat, lon, level, time)`  -> `(level, lat*lon*time)`

The sample ordering (lat fastest, then lon, then time) is the same for both cases,
so different variables and the target line up column-for-column.
"""
function flatten_pointwise(arr::AbstractArray)
    if ndims(arr) == 3
        nlat, nlon, nt = size(arr)
        return reshape(arr, 1, nlat * nlon * nt)
    elseif ndims(arr) == 4
        nlat, nlon, nlev, nt = size(arr)
        permuted = permutedims(arr, (3, 1, 2, 4))        # (level, lat, lon, time)
        return reshape(permuted, nlev, nlat * nlon * nt)
    else
        error("expected a 3D (lat,lon,time) or 4D (lat,lon,level,time) array, got $(ndims(arr))D")
    end
end

# ----------------------------------------------------------------------------
# Cleaning & normalisation
# ----------------------------------------------------------------------------

"Keep only the columns (samples) where every input AND target entry is finite (drops ocean/missing points)."
function drop_invalid(X::AbstractMatrix, Y::AbstractMatrix)
    valid = vec(all(isfinite, X; dims = 1)) .& vec(all(isfinite, Y; dims = 1))
    return X[:, valid], Y[:, valid]
end

"Per-feature (per-row) mean and standard deviation across all samples."
function feature_mean_std(X::AbstractMatrix)
    n = size(X, 2)
    μ = sum(X; dims = 2) ./ n
    σ = sqrt.(max.(sum(abs2, X .- μ; dims = 2) ./ max(n - 1, 1), 0))
    return vec(μ), vec(σ)
end

"Standardise rows of `X` with the given mean/std vectors (zero std -> left unscaled)."
function standardize(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector)
    σsafe = map(s -> iszero(s) ? one(s) : s, σ)
    return (X .- μ) ./ σsafe
end

# ----------------------------------------------------------------------------
# Assemble the full dataset
# ----------------------------------------------------------------------------

"""
    pointwise_dataset(inputs, target; normalize=true, drop_missing=true,
                      target_transform=identity)

Build the point-wise `(X, Y)` matrices plus the normalisation statistics.

Arguments
  * `inputs::Vector{NetCDFVar}` – the predictor variables.
  * `target::NetCDFVar`         – the target variable (e.g. surface roughness).

Keywords
  * `target_transform` – applied element-wise to the target before normalising,
    e.g. `log` if you regress in log space (the parameterization predicts
    `log` surface roughness).
  * `normalize`     – standardise inputs and target to zero mean / unit std.
  * `drop_missing`  – drop grid points with any `NaN`/missing input or target.

Returns a NamedTuple `(; X, Y, input_mean, input_std, target_mean, target_std)`,
with `X` of shape `(features, samples)` and `Y` of shape `(1, samples)`.
Store the returned stats to de-normalise predictions at inference time.
"""
function pointwise_dataset(inputs::AbstractVector{NetCDFVar}, target::NetCDFVar;
        normalize::Bool = true, drop_missing::Bool = true,
        target_transform = identity)

    X = reduce(vcat, (flatten_pointwise(load_field(v)) for v in inputs))
    Y = target_transform.(flatten_pointwise(load_field(target)))

    if drop_missing
        X, Y = drop_invalid(X, Y)
    end

    input_mean, input_std   = feature_mean_std(X)
    target_mean, target_std = feature_mean_std(Y)

    if normalize
        X = standardize(X, input_mean, input_std)
        Y = standardize(Y, target_mean, target_std)
    end

    return (; X, Y, input_mean, input_std, target_mean, target_std)
end

# ----------------------------------------------------------------------------
# MLUtils.DataLoader wrapping
# ----------------------------------------------------------------------------

"""
    pointwise_dataloaders(inputs, target; batchsize=1024, split=0.8,
                          shuffle=true, rng=Random.default_rng(), kwargs...)

Convenience wrapper that builds the dataset and returns
`(train_loader, val_loader, stats)`, where the loaders are `MLUtils.DataLoader`s
yielding `(x, y)` batches with `x :: (features, batchsize)` ready for Lux.

`split` is the fraction of samples used for training (set to `1.0` for no
validation split). Remaining `kwargs` are forwarded to `pointwise_dataset`
(`normalize`, `drop_missing`, `target_transform`).
"""
function pointwise_dataloaders(inputs::AbstractVector{NetCDFVar}, target::NetCDFVar;
        batchsize::Int = 1024, split::Real = 0.8, shuffle::Bool = true,
        rng::AbstractRNG = Random.default_rng(), kwargs...)

    data = pointwise_dataset(inputs, target; kwargs...)
    X, Y = data.X, data.Y

    (Xtrain, Ytrain), (Xval, Yval) = splitobs((X, Y); at = split, shuffle = true)

    train_loader = DataLoader((Xtrain, Ytrain); batchsize, shuffle, partial = false, rng)
    val_loader   = DataLoader((Xval, Yval); batchsize, shuffle = false, partial = true)

    stats = (; data.input_mean, data.input_std, data.target_mean, data.target_std)
    return train_loader, val_loader, stats
end

# ----------------------------------------------------------------------------
# Example usage (surface roughness, matching parameterization.jl)
# ----------------------------------------------------------------------------
#
#   inputs = [
#       NetCDFVar(path = "vegetation_high.nc", name = "cvh"),
#       NetCDFVar(path = "vegetation_low.nc",  name = "cvl"),
#       NetCDFVar(path = "geopotential.nc",    name = "z",    level = 1),  # surface level
#       NetCDFVar(path = "snow_depth.nc",      name = "sd"),
#       NetCDFVar(path = "soil.nc",            name = "stl",  level = 4),  # bottom soil layer
#       NetCDFVar(path = "soil.nc",            name = "swvl", level = 1),  # top soil layer
#   ]
#   target = NetCDFVar(path = "surface_roughness.nc", name = "fsr")
#
#   train_loader, val_loader, stats =
#       pointwise_dataloaders(inputs, target; batchsize = 2048, target_transform = log)
#
#   for (x, y) in train_loader
#       # x :: (features, batchsize), y :: (1, batchsize)  -> feed straight into Lux
#   end
#
# `stats.input_mean / input_std` and `stats.target_mean / target_std` are exactly
# the normalisation constants the LearnedSurfaceRoughness scheme needs at inference.
