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

using Lux, Optimisers, Zygote, ADTypes, Random, Printf, JLD2
using MLDataDevices: cpu_device, gpu_device
using Lux: Training

include("dataloaders.jl")

"""
    train_model(model, train_loader, val_loader=nothing; kwargs...)

Train `model` on the column-wise data with AdamW + (by default) MSE loss, in
whatever space the target was prepared in by the dataloader (e.g. normalised
`log` surface roughness).

Keywords
  * `epochs`, `learning_rate` – max number of epochs and AdamW learning rate.
  * `lossfn`           – any Lux loss with the `(model, ps, st, (x,y))` signature.
  * `ad`               – AD backend, `AutoZygote()` by default.
  * `device`           – `cpu_device()` (default) or `gpu_device()`.
  * `patience`         – early-stopping patience. Training stops once the
                         validation loss has not improved (by more than
                         `min_delta`) for `patience` consecutive epochs, and the
                         best-validation parameters/states are restored. Active
                         only when a `val_loader` is given and `patience > 0`.
  * `min_delta`        – minimum decrease in validation loss that counts as an
                         improvement.
  * `rng`, `verbose`.

Returns `(; train_state, ps, st, history, best_epoch, best_val)` where `ps`/`st`
are the trained parameters and the **test-mode** states (the best-validation ones
when early stopping is active, ready to drop into a Lux inference call or the
`LearnedSurfaceRoughness` scheme), and `history` holds per-epoch train/validation
losses.
"""
function train_model(model, train_loader, val_loader = nothing;
        epochs::Int = 50, learning_rate = 1.0f-3, lossfn = MSELoss(), ad = AutoZygote(),
        device = cpu_device(), patience::Int = 10, min_delta::Real = 0,
        rng::AbstractRNG = Random.default_rng(), verbose = true)

    ps, st = device(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, AdamW(learning_rate))

    history = (; train = Float64[], val = Float64[])

    early_stop = patience > 0 && !isnothing(val_loader)   # need validation to monitor
    best_val, best_epoch, stale = Inf, 0, 0               # early-stopping bookkeeping
    best_ps = best_st = nothing                           # snapshot of best weights

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

        # --- early stopping: track the best epoch and stop when patience runs out ---
        if early_stop
            if val_loss < best_val - min_delta
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_ps = deepcopy(train_state.parameters)
                best_st = deepcopy(train_state.states)
            else
                stale += 1
            end
            if stale >= patience
                verbose && @info "early stopping" stopped_at = epoch best_epoch best_val
                break
            end
        end
    end

    # restore the best-validation weights when early stopping was active
    out_ps = isnothing(best_ps) ? train_state.parameters : best_ps
    out_st = isnothing(best_st) ? train_state.states : best_st

    return (; train_state,
              ps = out_ps,
              st = Lux.testmode(out_st),
              history, best_epoch, best_val)
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
# Saving / loading the trained model
# ----------------------------------------------------------------------------

"""
    save_model(path, model, ps, st; stats=nothing)

Save a trained model to `path` (a `.jld2` file) with JLD2. Parameters and states
are moved to the CPU first so the file is portable across CPU/GPU runs. Pass the
dataloader `stats` (input/target mean & std) to bundle the normalisation
constants needed at inference. Returns `path`.
"""
function save_model(path::AbstractString, model, ps, st; stats = nothing)
    cdev = cpu_device()
    jldsave(path; model, parameters = cdev(ps), states = cdev(st), stats)
    return path
end

"""
    load_model(path) -> (; model, parameters, states, stats)

Load a model saved with [`save_model`](@ref). The same packages used to build the
model (e.g. Lux) must be loaded for deserialisation to succeed.
"""
function load_model(path::AbstractString)
    data = JLD2.load(path)
    return (; model = data["model"],
              parameters = data["parameters"],
              states = data["states"],
              stats = data["stats"])
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
#   result = train_model(model, train_loader, val_loader;
#                        epochs = 200, learning_rate = 1.0f-3, patience = 15)
#
#   # result.ps / result.st are the best-validation parameters & test-mode states.
#   # Bundle them with `stats` (input/target mean & std) — everything the
#   # LearnedSurfaceRoughness scheme needs at inference — and save:
#   save_model("surface_roughness_model.jld2", model, result.ps, result.st; stats)
#   saved = load_model("surface_roughness_model.jld2")
