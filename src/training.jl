"""
    train(params, dtrain; [num_boost_round = 10, evals = Vector{Tuple{DMatrix,String}}(),
          obj = nothing, feval = nothing, maximize = false, early_stopping_rounds = nothing,
          verbose_eval = true, xgb_model = nothing, callbacks = nothing])

Return a Booster trained with the given parameters.

# Arguments
* `params::Dict{String,<:Any}`: Booster params.
* `dtrain::DMatrix`: Data to be trained.
* `num_boost_round::Int`: Number of boosting iterations.
* `evals::Vector{Tuple{DMatrix,String}}`: Vector of items to be evaluated during training, this
    allows user to watch performance on the validation set.
* `obj::Union{Function,Void}`: Customized objective function.
* `feval::Union{Function,Void}`:  Customized evaluation function.
* `maximize::Bool`: Whether to maximize feval.
* `early_stopping_rounds::Union{Int,Void}`: Activates early stopping. Validation error needs to
    decrease at least every `early_stopping_rounds` round(s) to continue training. Requires at
    least one item in evals. If there’s more than one, will use the last.
* `verbose_eval::Union{Bool,Int}`: Requires at least one item in evals. If `verbose_eval` is `true`
    then the evaluation metric on the validation set is printed at each boosting stage. If
    `verbose_eval` is an integer then the evaluation metric on the validation set is printed at
    every given `verbose_eval` boosting stage. The last boosting stage / the boosting stage found
    by using early_stopping_rounds is also printed.
* `xgb_model::Union{Booster,String,Void}`: XGB model to be loaded before training (allows training
    continuation).
* `callbacks::Union{Vector{Function},Void}`: Vector of callback functions. Callback functions
    should have a field `cb_timing` that indicates when the callback should run. Can be "before",
    or "after" the training iteration.
"""
@compat function train(params::Dict{String,<:Any}, dtrain::DMatrix;
                       num_boost_round::Int = 10,
                       evals::Vector{Tuple{DMatrix,String}} = Vector{Tuple{DMatrix,String}}(),
                       obj::Union{Function,Void} = nothing, feval::Union{Function,Void} = nothing,
                       maximize::Bool = false, early_stopping_rounds::Union{Int,Void} = nothing,
                       verbose_eval::Union{Bool,Int} = true,
                       xgb_model::Union{Booster,String,Void} = nothing,
                       callbacks::Union{Vector{Function},Void} = nothing)

    callbacks_vec = isa(callbacks, Vector{Function}) ? callbacks : Vector{Function}()

    if isa(verbose_eval, Bool) && verbose_eval == true
        push!(callbacks_vec, cb_print_evaluation())
    elseif isa(verbose_eval, Int)
        push!(callbacks_vec, cb_print_evaluation(verbose_eval))
    end

    if !isa(early_stopping_rounds, Void)
        push!(callbacks_vec, cb_early_stop(early_stopping_rounds, maximize, verbose_eval, params,
                                           evals, feval))
    end

    # Initialize the Booster with the appropriate caches.
    if isa(xgb_model, Void)
        cache = unshift!([eval[1] for eval in evals], dtrain)
        bst = Booster(params = params, cache = cache)
        num_boost = 0
    else
        if isa(xgb_model, String)
            _xgb_model = xgb_model
        else
            _xgb_model = save_raw(xgb_model)
        end
        cache = unshift!([eval[1] for eval in evals], dtrain)
        bst = Booster(params = params, cache = cache;
                      model_file = _xgb_model)
        num_boost = length(get_dump(bst))
    end

    # TODO: Implement this functionality.
    if haskey(params, "num_parallel_tree")
        num_parallel_tree = params["num_parallel_tree"]
        num_boost /= num_parallel_tree
    else
        num_parallel_tree = 1
    end

    if haskey(params, "num_class")
        num_boost /= params["num_class"]
    end

    # Add distributed code support.
    rank = 0
    start_iteration = 1

    callbacks_before_iter = [cb for cb in callbacks_vec if cb.cb_timing == "before"]
    callbacks_after_iter = [cb for cb in callbacks_vec if cb.cb_timing == "after"]

    results = init_results(evals)
    env = CallbackEnv(bst, CVPack[], 0, start_iteration, num_boost_round, rank, results)

    for iter in start_iteration:num_boost_round
        env.iteration = iter

        foreach(cb -> cb(env), callbacks_before_iter)

        update(bst, dtrain, iter, fobj = obj)
        num_boost += 1
        # Add distributed code support.

        if length(evals) > 0
            evalstring = eval_set(bst, evals, iter, feval = feval)
            row = 1 + iter - start_iteration
            num_rows = 1 + num_boost_round - start_iteration
            insert_evals!(results, evalstring, row, 1, num_rows, 1)
        end

        try
            foreach(cb -> cb(env), callbacks_after_iter)
        catch err
            if isa(err, EarlyStopException)
                break
            else
                rethrow(err)
            end
        end
        # Add distributed code
    end

    # if attr(bst, "best_score") != ""
    #     bst.best_score = float(attr(bst, "best_score"))
    #     bst.best_iteration = parse(Int, attr(bst, "best_iteration"))
    # else
    #     bst.best_iteration = num_boost - 1
    # end
    # bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree

    return bst
end


# Return a results dictionary with a dictionary entry for each eval_name.
function init_results(evals::Vector{Tuple{DMatrix,String}})
    results = Dict(eval[2] => Dict{String,Matrix{Float64}}() for eval in evals)
    return results
end


# Insert the entries in the evalstring into the results dictionary.
function insert_evals!(results::Dict{String,Dict{String,Matrix{Float64}}}, evalstring::String,
                       row::Int, col::Int, num_rows::Int, num_cols::Int)
    split_eval_set = split(evalstring, ['\t', '-', ':'], keep = false)
    num_evals = div(length(split_eval_set) + 1, 3)

    for eval_idx in 1:num_evals
        eval_name_idx = 2 + 3 * (eval_idx - 1)
        eval_name = split_eval_set[eval_name_idx]
        metric_name = split_eval_set[eval_name_idx + 1]
        metric_value = split_eval_set[eval_name_idx + 2]

        curr_eval = results[eval_name]
        if !haskey(curr_eval, metric_name)
            curr_eval[metric_name] = Matrix{Float64}(num_rows, num_cols)
        end

        curr_eval[metric_name][row] = float(metric_value)
    end

    return results
end
