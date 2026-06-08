# Training a column-wise Lux model with `Lux.Training`.
#
# Pairs with dataloaders.jl. The DataLoaders there yield `(x, y)` batches with
#     x :: (features, batch)
#     y :: (1, batch)
#
# The training loop is the standard Lux.Training pattern:
#   1. `TrainState(model, ps, st, optimizer)` bundles params, states and optimiser.
#   2. `single_train_step!(ad, lossfn, (x, y), state)` does fwd + backward + update.
#   3. evaluate on a validation loader with the states put in test mode.

using Lux, Optimisers, Zygote, ADTypes, Random, Printf
using MLDataDevices: cpu_device, gpu_device
using Lux: Training 

include("dataloaders.jl")

"""
    train_model(model, train_loader, val_loader=nothing; kwargs...)

Train `model` on the column-wise data with AdamW + (by default) MSE loss, in
whatever space the target was prepared in by the dataloader (e.g. normalised
`log` surface roughness).

Keywords
  * `epochs`, `learning_rate` – number of epochs and AdamW learning rate.
  * `lossfn`           – any Lux loss with the `(model, ps, st, (x,y))` signature.
  * `ad`               – AD backend, `AutoZygote()` by default.
  * `device`           – `cpu_device()` (default) or `gpu_device()`.
  * `rng`, `verbose`.

Returns `(; train_state, ps, st, history)` where `ps`/`st` are the trained
parameters and the **test-mode** states (ready to drop into a Lux inference
call or the `LearnedSurfaceRoughness` scheme), and `history` holds per-epoch
train/validation losses.
"""
function train_model(model, train_loader, val_loader = nothing;
        epochs::Int = 50, learning_rate = 1.0f-3, lossfn = MSELoss(), ad = AutoZygote(),
        device = cpu_device(), rng::AbstractRNG = Random.default_rng(), verbose = true)

    ps, st = device(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, AdamW(learning_rate))

    history = (; train = Float64[], val = Float64[])

    for epoch in 1:epochs
        # --- one pass over the training data ---
        running, nbatches = 0.0, 0
        for (x, y) in train_loader
            x, y = device(x), device(y)
            _, loss, _, train_state =
                Training.single_train_step!(ad, lossfn, (x, y), train_state)
            running += loss
            nbatches += 1
        end
        train_loss = running / max(nbatches, 1)
        push!(history.train, train_loss)

        # --- validation (no parameter update) ---
        val_loss = isnothing(val_loader) ? NaN :
            evaluate(model, train_state.parameters, train_state.states, val_loader; lossfn, device)
        push!(history.val, val_loss)

        verbose && @printf("epoch %4d   train %.5f   val %.5f\n", epoch, train_loss, val_loss)
    end

    return (; train_state,
              ps = train_state.parameters,
              st = Lux.testmode(train_state.states),
              history)
end

"""
    evaluate(model, ps, st, loader; lossfn=MSELoss(), device=cpu_device())

Mean loss over `loader` without updating parameters. States are put in test mode
so layers like `Dropout` behave deterministically.
"""
function evaluate(model, ps, st, loader; lossfn = MSELoss(), device = cpu_device())
    st = Lux.testmode(st)
    running, nbatches = 0.0, 0
    for (x, y) in loader
        x, y = device(x), device(y)
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
#   # 1. preprocess NetCDF -> Zarr once (see preprocessing.jl):
#   #    preprocess_to_zarr(inputs, target, "surface_roughness.zarr")
#   #
#   # 2. build dataloaders from the Zarr store:
#   train_loader, val_loader, stats =
#       pointwise_dataloaders("surface_roughness.zarr"; batchsize = 2048, target_transform = log)
#
#   # hand over your own model (e.g. the Chain from LearnedSurfaceRoughness);
#   # its input width must match size(first(train_loader)[1], 1).
#   model  = Chain(Dense(7 => 32, leakyrelu), Dense(32 => 1))
#   result = train_model(model, train_loader, val_loader; epochs = 50, learning_rate = 1.0f-3)
#
#   # result.ps / result.st are the trained parameters & test-mode states.
#   # Together with `stats` (input/target mean & std) they are everything the
#   # LearnedSurfaceRoughness scheme needs at inference time.
