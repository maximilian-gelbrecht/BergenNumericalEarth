# Preprocessing: global NetCDF fields -> column-wise Zarr store prepared for dataloaders
#
#   1. Read the requested variables from one big multi-year NetCDF file, loading only
#      a single year's time-slice into RAM at a time. We work with land variables,
#      several of which carry missing values over the ocean.
#   2. Flatten every field column-wise: each grid point (at each time step) becomes
#      one sample / column.
#   3. Drop every sample that is NaN/missing in ANY input or the target, so that
#      the Zarr — and therefore the dataloaders — never see an invalid value.
#   4. Append each year's filtered, land-only columns to a per-variable Zarr array.

using NCDatasets, Zarr, Dates

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

"""
    load_var_slice(ds, name, tidx; level=nothing) -> Array{Float32}

Read variable `name` from the already-open dataset `ds`, restricted to time indices
`tidx` along the last dimension (`Colon()` loads all times). Only the requested
slices are read from disk. missings -> `NaN`; an optional `level` is selected.
"""
function load_var_slice(ds, name::AbstractString, tidx; level = nothing)
    var = ds[name]
    arr = ndims(var) == 3 ? var[:, :, tidx] :
          ndims(var) == 4 ? var[:, :, :, tidx] :
          error("expected a 3D (lat,lon,time) or 4D (lat,lon,level,time) variable '$(name)', got $(ndims(var))D")
    arr = to_float32_nan(arr)
    if level !== nothing && ndims(arr) == 4
        arr = arr[:, :, level, :]
    end
    return arr
end

"Map each calendar year to the time indices belonging to it (from the CF-decoded time coordinate)."
function year_indices(ds, time_name::AbstractString)
    haskey(ds, time_name) || error("time coordinate '$(time_name)' not found in dataset")
    times = ds[time_name][:]
    eltype(times) <: Dates.TimeType ||
        error("time coordinate '$(time_name)' is not CF-decoded to dates (eltype $(eltype(times))); cannot split by year")
    idx = Dict{Int, Vector{Int}}()
    for (i, t) in enumerate(times)
        push!(get!(idx, Int(Dates.year(t)), Int[]), i)
    end
    return idx
end

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
    preprocess_to_zarr(inputs, target, zarr_path; years=nothing, time_name="time",
                       chunk_samples=8192, overwrite=true, compressor=..., verbose=true)

Build the column-wise dataset from a big multi-year NetCDF file and persist it as a
Zarr group, loading only one year's time-slice into RAM at a time.

Arguments
  * `inputs::Vector{NetCDFVar}` – the predictor variables (land variables).
  * `target::NetCDFVar`         – the target variable (e.g. surface roughness).
  * `zarr_path::AbstractString` – directory store to create.

Memory model
  * `years` is the collection of calendar years to process (e.g. `2000:2020`). The
    CF-decoded `time_name` coordinate is used to find each year's time indices, and
    every variable is read sliced to those indices — so only one year of data is in
    memory at once. With `years=nothing` the whole file is processed in one go.
  * Each year is handled independently in a single pass: load all variables for the
    year, drop every sample that is missing in ANY of them (the join across
    variables), then append the surviving land-only columns to each variable's Zarr
    array. Masks may differ year to year — that is fine, since samples are stored
    column-wise.

Output
  * One array per variable, keyed by `NetCDFVar.name`, of shape `(features,
    n_samples)`, grown year by year, chunked along the sample dimension and
    compressed with `compressor`. Group attributes record `input_names` and
    `target_name`; `n_samples` is simply each array's width.

`compressor` controls on-disk compression (Blosc + zstd by default); pass `nothing`
to store uncompressed. `verbose` logs per-(variable, year) progress and timings.

Assumes all variables share the same `time` axis (same timestamps in the same
order — the usual case for one file).

Returns `(; path, input_names, target_name, n_samples, n_dropped, years)`.
"""
function preprocess_to_zarr(inputs::AbstractVector{<:NetCDFVar}, target::NetCDFVar,
        zarr_path::AbstractString; years = nothing, time_name::AbstractString = "time",
        chunk_samples::Int = 8192, overwrite::Bool = true,
        compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 5, shuffle = 1),
        verbose::Bool = true)

    vars  = NetCDFVar[inputs..., target]            # inputs first, target last
    names = String[v.name for v in vars]
    allunique(names) || error("variable names must be unique (they key the Zarr arrays); got $(names)")

    # open each distinct file once and keep the handles for the whole run
    datasets = Dict(p => NCDataset(p) for p in unique(v.path for v in vars))
    try
        # chunks = list of (label, path -> time indices). `nothing` years -> whole file.
        chunks = if isnothing(years)
            [("all", Dict(p => Colon() for p in keys(datasets)))]
        else
            file_year_idx = Dict(p => year_indices(datasets[p], time_name) for p in keys(datasets))
            [(string(y), Dict(p => get(file_year_idx[p], y, Int[]) for p in keys(datasets))) for y in years]
        end

        verbose && @info "Preprocessing → $(zarr_path)" variables=length(vars) years=(isnothing(years) ? "all" : length(chunks))

        # create the group; per-variable arrays are created lazily on first write,
        # taking their feature-row count from the loaded slice.
        if overwrite && ispath(zarr_path)
            rm(zarr_path; recursive = true, force = true)
        end
        g = zgroup(zarr_path; attrs = Dict(
            "input_names" => String[v.name for v in inputs],
            "target_name" => target.name,
        ))
        arrays = Dict{String, Any}()

        total_kept = total_seen = 0
        for (label, idxmap) in chunks
            # skip a requested year that is absent from any file
            if any(idxmap[v.path] isa AbstractVector && isempty(idxmap[v.path]) for v in vars)
                verbose && @warn "  year $(label): no data in at least one file, skipping"
                continue
            end

            # load every variable for this year and build the joint land mask
            fields = Dict{String, Any}()
            mask = nothing
            for v in vars
                t = time()
                f = flatten_pointwise(load_var_slice(datasets[v.path], v.name, idxmap[v.path]; level = v.level))
                fields[v.name] = f
                m = vec(all(isfinite, f; dims = 1))
                if isnothing(mask)
                    mask = m
                else
                    length(m) == length(mask) ||
                        error("sample-count mismatch in year $(label) for '$(v.name)': $(length(m)) vs $(length(mask))")
                    mask .&= m
                end
                verbose && @info "  $(label) · loaded $(v.name)" features=size(f, 1) samples=size(f, 2) seconds=round(time() - t; digits = 2)
            end

            nseen = length(mask)
            k = count(mask)
            total_seen += nseen
            total_kept += k

            # append this year's land-only, NaN-free columns to each variable's array
            # (created lazily on first write, with its feature-row count from the slice)
            if k > 0
                for v in vars
                    f = fields[v.name]
                    z = get!(arrays, v.name) do
                        zcreate(Float32, g, v.name, size(f, 1), 0;
                                chunks = (size(f, 1), chunk_samples), compressor = compressor)
                    end
                    old = size(z, 2)
                    resize!(z, size(z, 1), old + k)
                    z[:, old+1:old+k] = f[:, mask]
                end
            end
            verbose && @info "  $(label) · appended" kept=k dropped=(nseen - k) keep_fraction=round(k / max(nseen, 1); digits = 3)
        end

        total_kept > 0 || error("no valid land samples remain after filtering missings")
        verbose && @info "Preprocessing done" path=zarr_path n_samples=total_kept

        return (; path = zarr_path,
                  input_names = String[v.name for v in inputs],
                  target_name = target.name,
                  n_samples = total_kept,
                  n_dropped = total_seen - total_kept,
                  years = isnothing(years) ? nothing : collect(years))
    finally
        foreach(close, values(datasets))
    end
end