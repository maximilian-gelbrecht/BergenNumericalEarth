import Pkg 
Pkg.activate(".")

using SpeedyWeather, Lux 


include("parameterization.jl")



arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc=32, architecture=arch)
surface_roughness = LearnedSurfaceRoughness(spectral_grid)

model = PrimitiveWetModel(spectral_grid; surface_roughness)
simulation = initialize!(model)
run!(simulation, steps=2)