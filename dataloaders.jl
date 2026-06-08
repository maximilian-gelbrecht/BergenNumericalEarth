# Column-wise dataloaders for training Lux.jl models.
#
# Here we only assemble those arrays into model-ready batches:
#   * vertically stack the input variables into a feature matrix X,
#   * read the target into Y,
#   * (optionally) standardise, then split and wrap in `MLUtils.DataLoader`.
#
# Every batch comes out as `x :: (features, batch)`, `y :: (targets, batch)` —
# the trailing dimension is the batch axis Lux expects.

using Zarr, MLUtils, Random

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

"""
    load_zarr_dataset(zarr_path; normalize=true, target_transform=identity)

Read the column-wise dataset written by `preprocess_to_zarr`. Stacks the input
variables (in the stored `input_names` order) into `X :: (features, samples)` and
reads the target into `Y :: (targets, samples)`.

Keywords
  * `target_transform` – applied element-wise to the target before normalising,
    e.g. `log` to regress in log space (the parameterization predicts `log`
    surface roughness).
  * `normalize` – standardise inputs and target to zero mean / unit std.

Returns `(; X, Y, input_mean, input_std, target_mean, target_std, input_names, target_name)`.
The data is already clean (preprocessing dropped every NaN/missing sample).
"""
function load_zarr_dataset(zarr_path::AbstractString;
        normalize::Bool = true, target_transform = identity)

    g = zopen(zarr_path, "r")
    input_names = String.(g.attrs["input_names"])
    target_name = String(g.attrs["target_name"])

    X = reduce(vcat, (Float32.(g[name][:, :]) for name in input_names))
    Y = target_transform.(Float32.(g[target_name][:, :]))

    input_mean, input_std   = feature_mean_std(X)
    target_mean, target_std = feature_mean_std(Y)

    if normalize
        X = standardize(X, input_mean, input_std)
        Y = standardize(Y, target_mean, target_std)
    end

    return (; X, Y, input_mean, input_std, target_mean, target_std, input_names, target_name)
end

"""
    pointwise_dataloaders(zarr_path; batchsize=1024, split=0.8, shuffle=true,
                          rng=Random.default_rng(), kwargs...)

Build train/validation `MLUtils.DataLoader`s from the preprocessed Zarr store and
return `(train_loader, val_loader, stats)`. Batches are `(x, y)` with
`x :: (features, batchsize)` and `y :: (targets, batchsize)`, ready for Lux.

`split` is the fraction of samples used for training (`1.0` -> no validation
split; the second loader then iterates an empty set). Remaining `kwargs` go to
`load_zarr_dataset` (`normalize`, `target_transform`).

`stats.input_mean / input_std` and `stats.target_mean / target_std` are the
normalisation constants the `LearnedSurfaceRoughness` scheme needs at inference.
"""
function pointwise_dataloaders(zarr_path::AbstractString;
        batchsize::Int = 1024, split::Real = 0.8, shuffle::Bool = true,
        rng::AbstractRNG = Random.default_rng(), kwargs...)

    data = load_zarr_dataset(zarr_path; kwargs...)
    X, Y = data.X, data.Y

    (Xtrain, Ytrain), (Xval, Yval) = splitobs((X, Y); at = split, shuffle = true)

    train_loader = DataLoader((Xtrain, Ytrain); batchsize, shuffle, partial = false, rng)
    val_loader   = DataLoader((Xval, Yval); batchsize, shuffle = false, partial = true)

    stats = (; data.input_mean, data.input_std, data.target_mean, data.target_std,
               data.input_names, data.target_name)
    return train_loader, val_loader, stats
end

# ----------------------------------------------------------------------------
# Example usage (after running preprocess_to_zarr from preprocessing.jl)
# ----------------------------------------------------------------------------
#
#   train_loader, val_loader, stats =
#       pointwise_dataloaders("surface_roughness.zarr"; batchsize = 2048, target_transform = log)
#
#   for (x, y) in train_loader
#       # x :: (features, batchsize), y :: (1, batchsize)  -> feed straight into Lux
#   end
