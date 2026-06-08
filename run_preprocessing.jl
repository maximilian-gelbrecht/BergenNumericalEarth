#!/usr/bin/env -S julia +1.12 --project
# Preprocess a global NetCDF file into a column-wise, land-only Zarr store.
#
# Usage:
#   julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr]
#
#   <input.nc>     NetCDF file holding all input variables and the target.
#   [output.zarr]  output Zarr store (default: "surface_roughness.zarr").

import Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "preprocessing.jl"))

function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println("""
        Usage: julia +1.12 --project=. run_preprocessing.jl <input.nc> [output.zarr]

          <input.nc>     NetCDF file with all input variables and the target.
          [output.zarr]  output Zarr store (default: surface_roughness.zarr).
        """)
        return
    end

    file = args[1]
    zarr_path = length(args) >= 2 ? args[2] : "surface_roughness.zarr"
    isfile(file) || error("input NetCDF file not found: $(file)")

    inputs = [
        NetCDFVar(path = file, name = "cvh"),    # vegetation high
        NetCDFVar(path = file, name = "cvl"),    # vegetation low
        NetCDFVar(path = file, name = "z"),      # geopotential, surface
        NetCDFVar(path = file, name = "sd"),     # snow depth
        NetCDFVar(path = file, name = "stl1"),   # top soil temperature
        NetCDFVar(path = file, name = "swvl1"),  # top soil moisture
    ]
    target = NetCDFVar(path = file, name = "fsr")  # surface roughness

    info = preprocess_to_zarr(inputs, target, zarr_path)
    @info "preprocessed" info.path info.n_samples info.n_dropped
    return info
end

main(ARGS)
