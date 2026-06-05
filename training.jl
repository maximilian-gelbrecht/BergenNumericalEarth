# Training a point-wise Lux model with `Lux.Training`.
#
# Pairs with dataloaders.jl. The DataLoaders there yield `(x, y)` batches with
#     x :: (features, batch)
#     y :: (1, batch)
# which is exactly what a Lux model consumes (batch over the trailing dimension).
#
# The training loop is the standard Lux.Training pattern:
#   1. `TrainState(model, ps, st, optimizer)` bundles params, states and optimiser.
#   2. `single_train_step!(ad, lossfn, (x, y), state)` does fwd + backward + update.
#   3. evaluate on a validation loader with the states put in test mode.

using Lux, Optimisers, Zygote, ADTypes, Random, Printf
using MLDataDevices: cpu_device, gpu_device

include("dataloaders.jl")

const Training = Lux.Training

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

"""
    train_model(model, train_loader, val_loader=nothing; kwargs...)

Train `model` on the point-wise data with Adam + (by default) MSE loss, in
whatever space the target was prepared in by the dataloader (e.g. normalised
`log` surface roughness).

Keywords
  * `epochs`, `lr`     – number of epochs and Adam learning rate.
  * `lossfn`           – any Lux loss with the `(model, ps, st, (x,y))` signature.
  * `ad`               – AD backend, `AutoZygote()` by default.
  * `dev`              – `cpu_device()` (default) or `gpu_device()`.
  * `rng`, `verbose`.

Returns `(; train_state, ps, st, history)` where `ps`/`st` are the trained
parameters and the **test-mode** states (ready to drop into a Lux inference
call or the `LearnedSurfaceRoughness` scheme), and `history` holds per-epoch
train/validation losses.
"""
function train_model(model, train_loader, val_loader = nothing;
        epochs::Int = 50, lr = 1.0f-3, lossfn = MSELoss(), ad = AutoZygote(),
        dev = cpu_device(), rng::AbstractRNG = Random.default_rng(), verbose = true)

    ps, st = dev(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, Adam(lr))

    history = (; train = Float64[], val = Float64[])

    for epoch in 1:epochs
        # --- one pass over the training data ---
        running, nbatches = 0.0, 0
        for (x, y) in train_loader
            x, y = dev(x), dev(y)
            _, loss, _, train_state =
                Training.single_train_step!(ad, lossfn, (x, y), train_state)
            running += loss
            nbatches += 1
        end
        train_loss = running / max(nbatches, 1)
        push!(history.train, train_loss)

        # --- validation (no parameter update) ---
        val_loss = isnothing(val_loader) ? NaN :
            evaluate(model, train_state.parameters, train_state.states, val_loader; lossfn, dev)
        push!(history.val, val_loss)

        verbose && @printf("epoch %4d   train %.5f   val %.5f\n", epoch, train_loss, val_loss)
    end

    return (; train_state,
              ps = train_state.parameters,
              st = Lux.testmode(train_state.states),
              history)
end

"""
    evaluate(model, ps, st, loader; lossfn=MSELoss(), dev=cpu_device())

Mean loss over `loader` without updating parameters. States are put in test mode
so layers like `Dropout` behave deterministically.
"""
function evaluate(model, ps, st, loader; lossfn = MSELoss(), dev = cpu_device())
    st = Lux.testmode(st)
    running, nbatches = 0.0, 0
    for (x, y) in loader
        x, y = dev(x), dev(y)
        loss, _, _ = lossfn(model, ps, st, (x, y))
        running += loss
        nbatches += 1
    end
    return running / max(nbatches, 1)
end

# ----------------------------------------------------------------------------
# Example usage (end to end with dataloaders.jl)
# ----------------------------------------------------------------------------
#
#   inputs = [
#       NetCDFVar(path = "vegetation_high.nc", name = "cvh"),
#       NetCDFVar(path = "vegetation_low.nc",  name = "cvl"),
#       NetCDFVar(path = "geopotential.nc",    name = "z",    level = 1),
#       NetCDFVar(path = "snow_depth.nc",      name = "sd"),
#       NetCDFVar(path = "soil.nc",            name = "stl",  level = 4),
#       NetCDFVar(path = "soil.nc",            name = "swvl", level = 1),
#   ]
#   target = NetCDFVar(path = "surface_roughness.nc", name = "fsr")
#
#   train_loader, val_loader, stats =
#       pointwise_dataloaders(inputs, target; batchsize = 2048, target_transform = log)
#
#   # hand over your own model (e.g. the Chain from LearnedSurfaceRoughness);
#   # its input width must match size(first(train_loader)[1], 1).
#   model  = Chain(Dense(7 => 32, leakyrelu), Dense(32 => 1))
#   result = train_model(model, train_loader, val_loader; epochs = 50, lr = 1.0f-3)
#
#   # result.ps / result.st are the trained parameters & test-mode states.
#   # Together with `stats` (input/target mean & std) they are everything the
#   # LearnedSurfaceRoughness scheme needs at inference time.
