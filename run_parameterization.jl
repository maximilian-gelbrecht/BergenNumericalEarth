import Pkg 
Pkg.activate(".")

using SpeedyWeather, Lux
## using CUDA, cuDNN   # only needed for arch = SpeedyWeather.GPU()

include("parameterization.jl")

# # Run it in a CPU simulation 
#
#

arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc=32, architecture=arch)
surface_roughness = LearnedSurfaceRoughness(spectral_grid)
boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)
model = PrimitiveWetModel(spectral_grid; boundary_layer)
simulation = initialize!(model)
run!(simulation, steps=2)


# # Run it in a GPU simulation 

# overwrite parameterization dispatch?