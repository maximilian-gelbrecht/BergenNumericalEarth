import Pkg 
Pkg.activate(".")

using SpeedyWeather, Lux
using CUDA, cuDNN

include("parameterization.jl")

arch = SpeedyWeather.GPU()
spectral_grid = SpectralGrid(trunc=32, architecture=arch)
surface_roughness = LearnedSurfaceRoughness(spectral_grid)
boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)
model = PrimitiveWetModel(spectral_grid; boundary_layer)
simulation = initialize!(model)
run!(simulation, steps=2)