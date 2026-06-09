#!/usr/bin/env -S julia +1.12 --project
# Preprocess a big multi-year NetCDF file into a column-wise, land-only Zarr store.
#
# Usage:
#   julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr] [years]
#
#   <input.nc>     big NetCDF file holding all input variables and the target.
#   [output.zarr]  output Zarr store (default: "surface_roughness.zarr").
#   [years]        years to process, e.g. 2000:2020, 2000-2020 or 2000,2005,2010.
#                  Only one year is held in memory at a time. Omit to process the
#                  whole file at once.

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "preprocessing.jl"))

const USAGE = """
Usage: julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr] [years]

  <input.nc>     big NetCDF file with all input variables and the target.
  [output.zarr]  output Zarr store (default: surface_roughness.zarr).
  [years]        e.g. 2000:2020, 2000-2020 or 2000,2005,2010 (omit = whole file).
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

    # remaining args: an output path and/or a years spec, in any order
    zarr_path = "surface_roughness.zarr"
    years = nothing
    for a in args[2:end]
        looks_like_years(a) ? (years = parse_years(a)) : (zarr_path = a)
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

    info = preprocess_to_zarr(inputs, target, zarr_path; years)
    @info "preprocessed" info.path info.n_samples info.n_dropped
    return info
end

main(ARGS)
