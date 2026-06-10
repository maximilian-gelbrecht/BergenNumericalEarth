# # Running the learned parameterization in SpeedyWeather.jl
#
# After training the surface roughness MLP offline and defining
# the `LearnedSurfaceRoughness` parameterization, we can
# now plug the trained network into an actual SpeedyWeather simulation.

import Pkg
Pkg.activate(".")

using SpeedyWeather, Lux, JLD2
## using CUDA, cuDNN   # only needed for arch = SpeedyWeather.GPU()

include("parameterization.jl")

# ## Import the trained model
#
# `training.jl` saved everything we need with `save_model`: the Lux network, its
# trained parameters, the matching states in test mode (so dropout is switched
# off at inference), and the normalisation statistics of the dataloaders.

trained = JLD2.load("trained_model.jld2")
land_nn     = trained["model"]
land_params = trained["parameters"]
land_states = trained["states"]       ## test-mode states: dropout disabled
stats       = trained["stats"]

# ## Map the normalisation constants to the scheme's inputs
#
# The dataloader `stats` hold the input means and standard deviations as vectors
# ordered like `stats.input_names` (the ERA5 short names), while
# `LearnedSurfaceRoughness` expects NamedTuples keyed by the descriptive variable
# names used in `surface_roughness_land`. So we translate the names and rebuild
# the constants as NamedTuples:

era5_to_scheme = Dict(
    "bare_soil" => :bare_soil,
    "cvh"       => :vegetation_high,
    "cvl"       => :vegetation_low,
    "z"         => :geopotential,
    "sd"        => :snow_depth,
    "stl1"      => :soil_temperature,
    "swvl1"     => :soil_moisture,
)

scheme_names = Tuple(era5_to_scheme[name] for name in stats.input_names)
land_input_means = NamedTuple{scheme_names}(Tuple(stats.input_mean))
land_input_stds  = NamedTuple{scheme_names}(Tuple(stats.input_std))

# The network was trained to predict the *normalised log* surface roughness, so
# the scheme also needs the target mean/std to undo that transformation:

land_output_mean = stats.target_mean[1]
land_output_std  = stats.target_std[1]

# ## Run it in a CPU simulation
#
# Now we construct the scheme and hand it to the model via the `BoundaryLayer`.
# Everything else stays at its defaults — the default `LandModel` already carries
# the vegetation, snow, soil temperature and soil moisture variables our network
# uses as inputs.

arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc = 32, architecture = arch)

surface_roughness = LearnedSurfaceRoughness(
    spectral_grid, land_nn, land_params, land_states,
    land_input_means, land_input_stds;
    land_output_mean, land_output_std)

boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)
model = PrimitiveWetModel(spectral_grid; boundary_layer)
simulation = initialize!(model)
run!(simulation, steps = 2)

# As a quick sanity check, the range of the surface roughness our network
# predicted over land (in meters, after undoing the log and the normalisation;
# the zero minimum is simply the ocean points of the land field):

extrema(simulation.variables.parameterizations.land.surface_roughness)

# ## Outlook: running on the GPU
#
# The column-wise scheme above is not GPU-ready yet: all columns share a single
# `land_input_buffer` (a data race in a parallel kernel) and `Lux.apply` of a
# full `Chain` cannot be called from inside a GPU kernel. For the GPU we would
# instead evaluate the network batched over all grid points at once, as a global
# parameterization — that's the next step after this workshop.
