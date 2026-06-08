# Preprocessing: global NetCDF fields -> column-wise Zarr store prepared for dataloaders
#
#   1. Read the requested variables from one (or several) NetCDF files. We work
#      with land variables, several of which carry missing values over the ocean.
#   2. Flatten every field column-wise: each grid point (at each time step) becomes
#      one sample / column.
#   3. Drop every sample that is NaN/missing in ANY input or the target, so that
#      the Zarr — and therefore the dataloaders — never see an invalid value.
#   4. Write each variable's filtered array into a Zarr group, keyed by its name,
#      with the input/target bookkeeping stored as group attributes.

using NCDatasets, Zarr

"""
    NetCDFVar(; path, name, level=nothing)

Points at one variable `name` inside the NetCDF file at `path`.

`level` selects along the vertical dimension of a 4D (lat, lon, level, time) field:
  * `nothing`  – keep all levels (each level becomes a separate feature),
                 or no-op for a 3D field that has no level dimension,
  * `Integer`  – take a single level, e.g. a surface or bottom-soil index,
  * a range / vector of indices – keep a subset of levels.

`name` doubles as the key the variable is stored under in the Zarr group.
"""
Base.@kwdef struct NetCDFVar{L}
    path::String
    name::String
    level::L = nothing
end

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

"""
    preprocess_to_zarr(inputs, target, zarr_path; chunk_samples=8192, overwrite=true, compressor=...)

Build the column-wise dataset from NetCDF and persist it as a Zarr group.

Arguments
  * `inputs::Vector{NetCDFVar}` – the predictor variables (land variables).
  * `target::NetCDFVar`         – the target variable (e.g. surface roughness).
  * `zarr_path::AbstractString` – directory store to create.

What it does
  * Loads and flattens every variable to `(features, samples)`.
  * Builds one joint mask: a sample is kept only if every input AND the
    target are finite there. This removes ocean / missing points up front, so the
    Zarr contains land-only, NaN-free data.
  * Writes one array per variable, keyed by `NetCDFVar.name`, chunked along the
    sample dimension and compressed with `compressor`. Group attributes record
    `input_names`, `target_name` and `n_samples` so the dataloader is fully
    self-describing.

`compressor` controls on-disk compression (Blosc + zstd by default); pass
`nothing` to store uncompressed. Compression is applied per chunk, so `chunk_samples`
also sets the compression block granularity.

Returns a summary NamedTuple `(; path, input_names, target_name, n_samples, n_dropped)`.
"""
function preprocess_to_zarr(inputs::AbstractVector{<:NetCDFVar}, target::NetCDFVar,
        zarr_path::AbstractString; chunk_samples::Int = 8192, overwrite::Bool = true,
        compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 5, shuffle = 1),
        verbose::Bool = true)

    # name -> flattened (features, samples) matrix, inputs first then target
    vars = NetCDFVar[inputs..., target]
    names = String[v.name for v in vars]
    allunique(names) || error("variable names must be unique (they key the Zarr arrays); got $(names)")

    verbose && @info "Preprocessing → $(zarr_path): loading $(length(vars)) variables from NetCDF"
    fields = map(enumerate(vars)) do (i, v)
        t = time()
        f = flatten_pointwise(load_field(v))
        verbose && @info "  loaded [$i/$(length(vars))] $(v.name)" features=size(f, 1) samples=size(f, 2) seconds=round(time() - t; digits = 2)
        f
    end

    nsamples = size(first(fields), 2)
    all(f -> size(f, 2) == nsamples, fields) ||
        error("all variables must share the same lat×lon×time grid (sample count mismatch)")

    # Joint validity mask across every feature of every variable.
    valid = trues(nsamples)
    for f in fields
        valid .&= vec(all(isfinite, f; dims = 1))
    end
    nkept = count(valid)
    nkept > 0 || error("no valid land samples remain after filtering missings")
    verbose && @info "  filtered missing/ocean samples" kept=nkept dropped=(nsamples - nkept) keep_fraction=round(nkept / nsamples; digits = 3)

    if overwrite && ispath(zarr_path)
        rm(zarr_path; recursive = true, force = true)
    end

    g = zgroup(zarr_path; attrs = Dict(
        "input_names" => String[v.name for v in inputs],
        "target_name" => target.name,
        "n_samples"   => nkept,
    ))

    verbose && @info "  writing $(length(names)) compressed arrays to Zarr"
    for (i, (name, f)) in enumerate(zip(names, fields))
        clean = f[:, valid]                              # land-only, NaN-free
        t = time()
        z = zcreate(Float32, g, name, size(clean)...;
                    chunks = (size(clean, 1), min(chunk_samples, nkept)),
                    compressor = compressor)
        z[:, :] = clean
        verbose && @info "  wrote [$i/$(length(names))] $(name)" size=size(clean) seconds=round(time() - t; digits = 2)
    end
    verbose && @info "Preprocessing done" path=zarr_path n_samples=nkept

    return (; path = zarr_path,
              input_names = String[v.name for v in inputs],
              target_name = target.name,
              n_samples = nkept,
              n_dropped = nsamples - nkept)
end

# This file is a library. To run preprocessing from the command line, use the
# companion script run_preprocessing.jl, e.g.
#   julia +1.12 --project=. run_preprocessing.jl path/to/era5_land.nc