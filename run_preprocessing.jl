#!/usr/bin/env -S julia +1.12 --project
# Preprocess a big multi-year NetCDF file into a column-wise, land-only Zarr store.
#
# Usage:
#   julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr] [years] [stride=N]
#
#   <input.nc>     big NetCDF file holding all input variables and the target.
#   [output.zarr]  output Zarr store (default: "surface_roughness.zarr").
#   [years]        years to process, e.g. 2000:2020, 2000-2020 or 2000,2005,2010.
#                  Only one year is held in memory at a time. Omit to process the
#                  whole file at once.
#   [stride=N]     keep every N-th time step (default 1 = all). The surface fields
#                  are highly redundant in time; for 3-hourly data 8≈daily,
#                  56≈weekly, 240≈monthly. Cuts samples (and chunk files) by N.
#   [chunk=N]      Zarr chunk width along samples (default 2^22 ≈ 4.2M, ~16 MiB).
#                  files ≈ (samples / chunk) × n_vars.
#   [lsm=PATH]     NetCDF file with a static land-sea mask variable named `lsm`
#                  (0–1 land fraction, same grid); ocean points (lsm < 0.5) are
#                  dropped. Without it nothing is dropped — ERA5 fills the ocean.

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "preprocessing.jl"))

const USAGE = """
Usage: julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr] [years] [stride=N]

  <input.nc>     big NetCDF file with all input variables and the target.
  [output.zarr]  output Zarr store (default: surface_roughness.zarr).
  [years]        e.g. 2000:2020, 2000-2020 or 2000,2005,2010 (omit = whole file).
  [stride=N]     keep every N-th time step (default 1); 3-hourly: 8≈daily, 240≈monthly.
  [chunk=N]      Zarr chunk width along samples (default 2^22 ≈ 4.2M, ~16 MiB).
  [lsm=PATH]     NetCDF with a static land-sea mask var `lsm` (same grid); drops ocean.
"""

"Does an argument look like a year range/list (e.g. 2000:2020, 2000-2020, 2000,2005)?"
looks_like_years(s) = occursin(r"^\d{3,4}([:\-,]\d{3,4})*$", s)

"Parse a year spec into a range or vector of Ints."
function parse_years(s)
    if occursin(':', s) || occursin('-', s)
        sep = occursin(':', s) ? ':' : '-'
        a, b = split(s, sep)
        return parse(Int, a):parse(Int, b)
    elseif occursin(',', s)
        return parse.(Int, split(s, ','))
    else
        return [parse(Int, s)]
    end
end

function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println(USAGE)
        return
    end

    file = args[1]
    isfile(file) || error("input NetCDF file not found: $(file)")

    # remaining args: output path, years spec, stride=N, chunk=N, lsm=PATH, any order
    zarr_path = "surface_roughness.zarr"
    years = nothing
    time_stride = 1
    chunk_samples = nothing            # nothing -> use preprocess_to_zarr's default
    landmask = nothing
    for a in args[2:end]
        if (m = match(r"^stride=(\d+)$", a)) !== nothing
            time_stride = parse(Int, m.captures[1])
        elseif (m = match(r"^chunk=(\d+)$", a)) !== nothing
            chunk_samples = parse(Int, m.captures[1])
        elseif (m = match(r"^lsm=(.+)$", a)) !== nothing
            lsm_path = String(m.captures[1])
            isfile(lsm_path) || error("land-sea mask file not found: $(lsm_path)")
            landmask = NetCDFVar(path = lsm_path, name = "lsm")
        elseif looks_like_years(a)
            years = parse_years(a)
        else
            zarr_path = a
        end
    end

    inputs = [
        NetCDFVar(path = file, name = "cvh"),    # vegetation high
        NetCDFVar(path = file, name = "cvl"),    # vegetation low
        NetCDFVar(path = file, name = "z"),      # geopotential, surface
        NetCDFVar(path = file, name = "sd"),     # snow depth
        NetCDFVar(path = file, name = "stl1"),   # top soil temperature
        NetCDFVar(path = file, name = "swvl1"),  # top soil moisture
    ]
    target = NetCDFVar(path = file, name = "fsr")  # surface roughness

    kw = (; years, time_stride)
    chunk_samples === nothing || (kw = (; kw..., chunk_samples))
    landmask === nothing || (kw = (; kw..., landmask))
    info = preprocess_to_zarr(inputs, target, zarr_path; kw...)
    @info "preprocessed" info.path info.n_samples info.n_dropped
    return info
end

main(ARGS)
