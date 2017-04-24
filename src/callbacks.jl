type CallbackEnv
    model::Booster
    cvfolds::Vector{CVPack}
    iteration::Int
    begin_iteration::Int
    end_iteration::Int
    rank::Int
    evaluation_results::Dict{String,Dict{String,Vector{Float64}}}
end


type EarlyStopException <: Exception
    best_iteration::Int
end


function cb_print_evaluation(period::Int = 1, show_stdv::Bool = true)
    cb_timing = "after"

    function callback(env::CallbackEnv)
        if env.rank != 0 || length(evaluation_result_list) == 0 || period == false
            return nothing
        end
        i = env.iteration
        if (i - 1) % period == 0 || i == env.begin_iteration || i == env.end_iteration
            # TODO: Add print script here
        end

        return nothing
    end

    return callback
end


function cb_early_stop(env::CallbackEnv, early_stopping_rounds::Integer,
                       early_stopping_metric::String, maximize::Bool = false, verbose = true)
    cb_timing = "after"

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
