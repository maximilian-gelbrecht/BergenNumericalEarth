# # Generalisability: the learned roughness on Planet Pangaea
#
# The strongest argument for learning the surface roughness as a *column-based*
# function of local surface variables — rather than hard-coding another boundary
# condition map — is that it generalises: the network doesn't know anything about
# Earth's geography, only about the relation between a column's surface state and
# its roughness. So nothing stops us from applying it to a planet whose geography
# the network has never seen.
#
# Here we do exactly that: we repeat the online simulation of
# [`run_parameterization.jl`](run_parameterization.jl), but on **Pangaea**, using the
# boundary conditions provided by
# [PlanetPangaea.jl](https://github.com/SpeedyWeather/PlanetPangaea.jl) —
# "climate modelling for dinosaurs". The package ships a full set of
# SpeedyWeather-ready NetCDF files (generated from an Olenekian Pangaea
# reconstruction image): land-sea mask, orography, high/low vegetation cover,
# a 12-month soil-moisture climatology, a 12-month SST climatology, and albedo.
# Conveniently they use exactly the variable names and grid layout that
# SpeedyWeather's boundary-condition components expect.

import Pkg
Pkg.activate(".")

using SpeedyWeather, Lux, JLD2

include("parameterization.jl")

# ## Get the Pangaea boundary conditions
#
# We download the six NetCDF files from the PlanetPangaea.jl repository (only once
# — files already on disk are kept). `Downloads` is a Julia standard library.

using Downloads

pangaea_dir = "pangaea"
pangaea_url = "https://raw.githubusercontent.com/SpeedyWeather/PlanetPangaea.jl/main/data/boundary_conditions"
pangaea_files = ("lsm.nc", "orography.nc", "vegetation.nc", "soil_moisture.nc", "sst.nc", "albedo.nc")

mkpath(pangaea_dir)
for file in pangaea_files
    local_path = joinpath(pangaea_dir, file)
    isfile(local_path) || Downloads.download(join((pangaea_url, file), "/"), local_path)
end
readdir(pangaea_dir)

# ## Fill the masked seasonal fields
#
# PlanetPangaea masks its seasonal fields with `NaN`: the soil moisture is `NaN`
# over the ocean and the SST is `NaN` over land. SpeedyWeather's climatology
# components interpolate these files onto the model grid as they are — and around
# the coasts that smears `NaN` into cells with *fractional* land. There the learned
# roughness would read `NaN` soil moisture, and the surface heat flux would read
# `NaN` SST: one contaminated coastal cell is enough to blow up the whole
# simulation within a time step (SpeedyWeather's own asset files are gap-free over
# the coasts for exactly this reason).
#
# So we fill the masked regions once with their nearest valid values — the same
# trick PlanetPangaea's generator uses internally before it re-masks. The filled
# values only ever matter in coastal cells: fluxes are weighted by the land
# fraction, so over pure ocean/land they have no effect.

using NCDatasets

"Replace NaNs by the mean of their finite neighbours, iterated until none are left (lon wraps around)."
function fill_nans!(A::AbstractMatrix)
    nlon, nlat = size(A)
    while any(isnan, A)
        B = copy(A)
        for j in 1:nlat, i in 1:nlon
            if isnan(A[i, j])
                s = 0.0f0; n = 0
                for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    ii = mod1(i + di, nlon)              ## periodic in longitude
                    jj = clamp(j + dj, 1, nlat)
                    if !isnan(A[ii, jj]); s += A[ii, jj]; n += 1; end
                end
                n > 0 && (B[i, j] = s / n)
            end
        end
        A .= B
    end
    return A
end

"Copy `src` to `dst` (once) and fill the NaNs of all `varnames`, month by month."
function fill_file(src, dst, varnames)
    isfile(dst) && return dst
    cp(src, dst)
    NCDataset(dst, "a") do ds
        for var in varnames, t in 1:ds.dim["time"]
            slice = Float32.(coalesce.(ds[var][:, :, t], NaN32))
            ds[var][:, :, t] = fill_nans!(slice)
        end
    end
    return dst
end

soil_moisture_path = fill_file(joinpath(pangaea_dir, "soil_moisture.nc"),
    joinpath(pangaea_dir, "soil_moisture_filled.nc"), ("swl1", "swl2"))
sst_path = fill_file(joinpath(pangaea_dir, "sst.nc"),
    joinpath(pangaea_dir, "sst_filled.nc"), ("sst",))

# ## Import the trained model
#
# Exactly as in [`run_parameterization.jl`](run_parameterization.jl): `training.jl`
# saved the Lux network, its trained parameters, the test-mode states and the
# normalisation statistics of the dataloaders. Note that the network was trained
# purely on ERA5 — present-day Earth — and will now be asked about a Triassic
# supercontinent.

trained = JLD2.load("trained_model.jld2")
land_nn     = trained["model"]
land_params = trained["parameters"]
land_states = trained["states"]       ## test-mode states: dropout disabled
stats       = trained["stats"]

# Map the normalisation constants from the ERA5 short names to the descriptive
# names the scheme uses, as before:

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

land_output_mean = stats.target_mean[1]
land_output_std  = stats.target_std[1]

# ## Swap Earth for Pangaea
#
# Every boundary-condition component in SpeedyWeather follows the same pattern: it
# reads a NetCDF variable from `path`, by default from the SpeedyWeatherAssets
# repository (`from_assets = true`). Pointing `path` at a local file and setting
# `from_assets = false` replaces Earth with whatever planet the file describes.
#
# The PlanetPangaea files use the standard variable names (`lsm`, `orog`,
# `vegh`/`vegl`, `swl1`/`swl2`, `sst`, `alb`) on a regular 384×192 longitude–latitude
# grid, which matches the layout of a full Gaussian grid with 192 rings — so the
# components' default `FullGaussianField` wrapping applies. Only the land-sea mask
# needs its `FieldType` overridden (its default expects the Earth asset file, which
# comes on a different grid).

arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc = 32, architecture = arch)

orography = EarthOrography(spectral_grid,
    path = joinpath(pangaea_dir, "orography.nc"), from_assets = false)

land_sea_mask = EarthLandSeaMask(spectral_grid,
    path = joinpath(pangaea_dir, "lsm.nc"), from_assets = false,
    FieldType = FullGaussianField)

ocean = SeasonalOceanClimatology(spectral_grid,
    path = sst_path, from_assets = false)

albedo = AlbedoClimatology(spectral_grid,
    path = joinpath(pangaea_dir, "albedo.nc"), from_assets = false)

# The land surface needs the vegetation cover (a direct input of our network!) and
# the seasonal soil moisture; both are components of the `LandModel`. Snow depth and
# soil temperature stay prognostic, as on Earth.

vegetation = VegetationClimatology(spectral_grid,
    path = joinpath(pangaea_dir, "vegetation.nc"), from_assets = false)

soil_moisture = SeasonalSoilMoisture(spectral_grid,
    path = soil_moisture_path, from_assets = false)

land = LandModel(spectral_grid; vegetation, soil_moisture)

# ## Run Pangaea with the learned surface roughness
#
# The learned scheme is constructed exactly as on Earth — it is the same network,
# the same weights, the same normalisation. Only the planet underneath changes.

surface_roughness = LearnedSurfaceRoughness(
    spectral_grid, land_nn, land_params, land_states,
    land_input_means, land_input_stds;
    land_output_mean, land_output_std)

model = PrimitiveWetModel(spectral_grid;
    orography, land_sea_mask, ocean, albedo, land, surface_roughness)

simulation = initialize!(model)
run!(simulation, period = Day(20))

# The range of the predicted roughness over land (in meters; the zero minimum is
# the ocean part of the land field):

extrema(simulation.variables.parameterizations.land.surface_roughness)

# ## Roughness of a supercontinent
#
# For the visual check we plot the learned surface roughness next to the high
# vegetation cover — the network's dominant predictor — interpolated to a regular
# grid. The roughness should trace the (fictional) vegetation belts and highlands
# of Pangaea, even though the network never saw a continent like this.

using CairoMakie

## helper: SpeedyWeather field -> (matrix, lon, lat) on its full regular grid
function to_lonlat(field)
    full = RingGrids.interpolate(RingGrids.full_grid_type(field.grid), field.grid.nlat_half, field)
    return Matrix(full), RingGrids.get_lond(full), RingGrids.get_latd(full)
end

z₀_sim, lond, latd = to_lonlat(simulation.variables.parameterizations.land.surface_roughness)
vegh_sim, _, _     = to_lonlat(model.land.vegetation.high_cover)

fig = Figure(size = (800, 650))
ax1 = Axis(fig[1, 1]; title = "Learned surface roughness on Pangaea (T32)")
hm1 = heatmap!(ax1, lond, latd, log10.(max.(z₀_sim, 1f-6)); colorrange = (-4, 0.5))
Colorbar(fig[1, 2], hm1, label = "log₁₀ z₀ [m]")
ax2 = Axis(fig[2, 1]; title = "High vegetation cover (PlanetPangaea)", xlabel = "longitude")
hm2 = heatmap!(ax2, lond, latd, vegh_sim; colorrange = (0, 1), colormap = :Greens)
Colorbar(fig[2, 2], hm2, label = "cover fraction")
CairoMakie.save("surface_roughness_pangaea.png", fig)   ## qualified: JLD2 also exports `save`
fig

# Spatial generalisation in action: a parameterization learned from ERA5 maps,
# applied to a planet that hasn't existed for 250 million years — because the
# network learned the *function*, not the map.
