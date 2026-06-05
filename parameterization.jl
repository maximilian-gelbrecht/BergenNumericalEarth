using Adapt, Random

@kwdef struct LearnedSurfaceRoughness{NF, V, M, LNN, LP, LS} <: SpeedyWeather.AbstractSurfaceRoughness
    "[OPTION] constant roughness length over ocean [m]"
    roughness_length_ocean::NF = 1.0e-4

    # learned land roughness parameters
    # Land normalisation parameters (NamedTuples keyed by the inputs used in
    # surface_roughness_land), built in the outer constructor below
    land_input_means::M
    land_input_stds::M

    land_output_mean::NF = -5.031811f0
    land_output_std::NF = 2.4447718f0

    # input buffer for NN input
    land_input_buffer::V

    # NN structure, parameters and states
    land_nn::LNN
    land_params::LP
    land_states::LS
end

function LearnedSurfaceRoughness(
        SG::SpectralGrid,
        land_nn = nothing, 
        land_params = nothing, 
        land_states = nothing;
        kwargs...
    )

    # Set up Lux NN, if it's not provided
    if isnothing(land_nn) 
        land_nn = Lux.Chain(
            Lux.Dense(7 => 32, Lux.leakyrelu),
            Lux.Dense(32 => 64, Lux.leakyrelu),
            Lux.Dropout(0.2),
            Lux.Dense(64 => 64, Lux.leakyrelu),
            Lux.Dropout(0.1),
            Lux.Dense(64 => 32, Lux.leakyrelu),
            Lux.Dense(32 => 1)
        )

        rng = Random.default_rng()
        land_params, rand_states = Lux.setup(rng, land_nn)
        land_states = Lux.testmode(rand_states)
    end

    land_input_buffer = on_architecture(SG.architecture, zeros(Float32, 7))

    # Land normalisation parameters, keyed by the inputs used in surface_roughness_land
    land_input_means = (
        bare_soil        = 7.4566591f-1,
        vegetation_high  = 1.0025085f-1,
        vegetation_low   = 1.5397815f-1,
        geopotential     = 1.6788273f+4,
        snow_depth       = 6.3441253f+0,
        soil_temperature = 2.5690454f+2,
        soil_moisture    = 2.1826939f-1,
    )
    land_input_stds = (
        bare_soil        = 4.23785776f-1,
        vegetation_high  = 2.76675612f-1,
        vegetation_low   = 3.28937173f-1,
        geopotential     = 1.16441348f+4,
        snow_depth       = 4.8089776f+0,
        soil_temperature = 3.01283646f+1,
        soil_moisture    = 1.26479045f-1,
    )

    return LearnedSurfaceRoughness{
        SG.NF,
        typeof(land_input_buffer),
        typeof(land_input_means),
        typeof(land_nn),
        typeof(land_params),
        typeof(land_states),
    }(;
        land_input_means = land_input_means,
        land_input_stds = land_input_stds,
        land_input_buffer = land_input_buffer,
        land_nn = land_nn,
        land_params = land_params,
        land_states = land_states,
        kwargs...
    )
end

Adapt.@adapt_structure LearnedSurfaceRoughness

SpeedyWeather.initialize!(::LearnedSurfaceRoughness, ::PrimitiveEquation) =  nothing

@inline function normalise(a, m, s)
    return (a - m) / s
end

Base.@propagate_inbounds function surface_roughness_land(ij, vars, scheme::LearnedSurfaceRoughness)
    
    vₕ = vars.parameterizations.land.vegetation_high[ij]
    vₗ =  vars.parameterizations.land.vegetation_low[ij]
    vᵦ = 1 - vₕ - vₗ  # bare soil
    g = vars.grid.geopotential[ij, end]
    sd = vars.prognostic.land.snow_depth[ij]
    soil_moisture = vars.prognostic.land.soil_moisture[ij, begin]  # currently top layer
    soil_temperature = vars.prognostic.land.soil_temperature[ij, end]  # currently bottom layer

    # Normalise inputs
    vₕ = normalise(vₕ, scheme.land_input_means.vegetation_high, scheme.land_input_stds.vegetation_high)
    vₗ = normalise(vₗ, scheme.land_input_means.vegetation_low, scheme.land_input_stds.vegetation_low)
    vᵦ = normalise(vᵦ, scheme.land_input_means.bare_soil, scheme.land_input_stds.bare_soil)
    g = normalise(g, scheme.land_input_means.geopotential, scheme.land_input_stds.geopotential)
    sd = normalise(sd, scheme.land_input_means.snow_depth, scheme.land_input_stds.snow_depth)
    soil_moisture = normalise(soil_moisture, scheme.land_input_means.soil_moisture, scheme.land_input_stds.soil_moisture)
    soil_temperature = normalise(soil_temperature, scheme.land_input_means.soil_temperature, scheme.land_input_stds.soil_temperature)

    scheme.land_input_buffer[:] .= (vᵦ, vₕ, vₗ, g, sd, soil_temperature, soil_moisture)

    prediction, _ = Lux.apply(scheme.land_nn, scheme.land_input_buffer, scheme.land_params, scheme.land_states)
    log_surface_roughness = (prediction[1] * scheme.land_output_std) + scheme.land_output_mean
    surface_roughness = exp(log_surface_roughness)
    return surface_roughness
end

Base.@propagate_inbounds function SpeedyWeather.surface_roughness!(ij, vars, scheme::LearnedSurfaceRoughness, land_sea_mask)
    land_fraction = land_sea_mask.mask[ij]

    # Compute separate ocean and land surface roughness
    vars.parameterizations.ocean.surface_roughness[ij] = ifelse(land_fraction > 0, scheme.roughness_length_ocean, zero(land_fraction)) 
    vars.parameterizations.land.surface_roughness[ij] = ifelse(land_fraction < 1, surface_roughness_land(ij, vars, scheme), zero(land_fraction)) 

    # Blend the two via arithmetic average
    vars.parameterizations.surface_roughness[ij] = land_fraction * vars.parameterizations.land.surface_roughness[ij] + (1 - land_fraction) * vars.parameterizations.ocean.surface_roughness[ij]
    return nothing
end
