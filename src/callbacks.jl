type CallbackEnv
    model::Booster
    cvfolds::Vector{CVPack}
    iteration::Int
    begin_iteration::Int
    end_iteration::Int
    rank::Int
    evaluation_results::Dict{String,Matrix{Float64}}
end

type EarlyStopException <: Exception
    best_iteration::Int
end


function cb_print_evalution(; period::Int = 1, show_stdv::Bool = true)

    function callback(env::CallbackEnv)
        if env.rank != 0 || length(evaluation_result_list) == 0 || period == false
            return nothing
        end
        i = env.iteration
        if i % period == 0 || i + 1 == env.begin_iteration
            # TODO: Add print script here
        end

        return nothing
    end

    return callback
end


function cb_record_evaluation(eval_result)
    for key in keys(eval_result)
        delete!(eval_result, key)
    end

    function init(env)
        for k in keys(env.evaluation_result_list)
            key, metric = split(k, '-')
            if !in(key, eval_result)
                eval_result[key] = # TODO: Add the eval_result type here
            end
            if !in(metric) in eval_result[key]
                eval_result[key][metric] = # TODO: Add the eval_result type here
            end
        end
    end

    function callback(env::CallbackEnv)
        if length(eval_result) == 0
            init(env) # TODO: Check that this form of initializing works correctly.
        end
        for (k, v) in env.evaluation_result_list
            key, metric = split(k, '-')
            eval_result[key][metric] # TODO: make sure v is added to this result list
        end
    end

    return callback
end


function cb_early_stop(env::CallbackEnv, early_stopping_rounds::Integer,
                       early_stopping_metric::String, maximize::Bool = false, verbose = true)
    if length(env.evaluation_result_list) == 0
        error("For early stopping you need at least one set in evals.")
    end

    maximize_metrics = ("-auc", "-map", "-ndcg")
    if any(i -> endswith(early_stopping_metric, i), maximize_metrics)
        maximize = true
    end

    maximize_at_n_metrics = ("auc@", "map@", "ndcg@")
    if any(i -> contains(early_stopping_metric, i), maximize_at_n_metrics)
        maximize = true
    end

    state_int = Dict{String,Int}
    state_float = Dict{String,Float}

    state_int["best_iteration"] = 0
    if maximize
        state_float["best_score"] = -Inf
    else
        state_float["best_score"] = Inf
    end

    function callback(env::CallbackEnv)
        if length(env.evaluation_results) == 0
            error("for early stopping you need at least one set in evals.")
        elseif !haskey(env.evaluation_results, early_stopping_metric)
            println("could not find early_stopping_metric ", early_stopping_metric,
                    "in evaluation_results")
        end

        iteration = env.iteration
        best_score = state_float["best_score"]
        best_iteration = state_int["best_iteration"]
        score = env.evaluation_results[early_stopping_metric][iteration, 1]

        if (maximize && score > best_score) || (!maximize && score < best_score)
            state_float["best_score"] = score
            state_int["best_iteration"] = iteration
            set_attr(env.model,
                     best_score = string(score),
                     best_iteration = string(iteration))
        elseif iteration - best_iteration >= early_stopping_rounds
            if verbose
                println("Stopping. Best iteration: ", best_iteration, ", ", early_stopping_metric,
                        ": ", best_score)
            end
            throw(EarlyStopException(best_iteration))
        end
    end

    return callback
end
