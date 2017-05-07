const maximize_metrics = ("auc", "map", "ndcg")
const maximize_at_n_metrics = ("auc@", "map@", "ndcg@")


type CallbackEnv
    model::Booster
    cvfolds::Vector{CVPack}
    iteration::Int
    begin_iteration::Int
    end_iteration::Int
    rank::Int
    results::Dict{String,Dict{String,Matrix{Float64}}}
end


type EarlyStopException <: Exception
    best_iteration::Int
end


function cb_print_evaluation(period::Int = 1, show_stdv::Bool = true)
    cb_timing = "after"

    function callback(env::CallbackEnv)
        cb_timing

        if env.rank != 0 || length(env.results) == 0
            return nothing
        end

        iter = env.iteration
        cur_results_idx = 1 + iter - env.begin_iteration
        if (iter - 1) % period == 0 || iter == env.begin_iteration || iter == env.end_iteration
            print("[", iter, "]")
            results = env.results
            for eval_name in keys(results)
                eval_results = results[eval_name]
                for eval_metric in keys(eval_results)
                    eval_metric_results = eval_results[eval_metric]
                    num_cols = size(eval_metric_results, 2)
                    if num_cols > 1
                        metric_values = eval_metric_results[cur_results_idx, :]
                        mean_metric_value = mean(metric_values)
                        print("\t", eval_name, "-", eval_metric, ":", mean_metric_value)
                        if show_stdv
                            std_metric_value = std(metric_values)
                            print("+", std_metric_value)
                        end
                    else
                        metric_value = eval_metric_results[cur_results_idx, 1]
                        print("\t", eval_name, "-", eval_metric, ":", metric_value)
                    end
                end
            end
            print("\n")
        end

        return nothing
    end

    return callback
end


function cb_early_stop(early_stopping_rounds::Integer, maximize::Bool,
                       verbose_eval::Union{Bool,Int}, params::Dict{String,<:Any},
                       evals::Vector{Tuple{DMatrix,String}}, feval::Union{Function,Void})
    cb_timing = "after"
    cb_early_stopping_metric = [""]
    cb_best_iteration = [0]
    cb_best_score = [Inf]
    cb_maximize = [maximize]

    # Establish the eval_name to use for early_stopping.
    if length(evals) == 0
        error("For early stopping you need to provide at least one one set in evals.")
    else
        cb_early_stopping_eval = evals[end][2]
    end

    # Establish the metric_name to use for early_stopping.
    if isa(feval, Function)
        if verbose_eval > 0 # For distributed version, add `&& env.rank == 0`.
            println("Will train until ", cb_early_stopping_eval, "'s feval hasn't improved in ",
                    early_stopping_rounds, " round(s).")
        end
    else
        if haskey(params, "eval_metric")
            params_eval_metric = params["eval_metric"]
            if isa(params_eval_metric, Array)
                cb_early_stopping_metric[1] = string(params_eval_metric[end])
            else
                cb_early_stopping_metric[1] = string(params_eval_metric)
            end
            if should_maximize(cb_early_stopping_metric[1])
                cb_maximize[1] = true
                cb_best_score[1] = -Inf
            else
                cb_maximize[1] = false
                cb_best_score[1] = Inf
            end
            if verbose_eval > 0 # For distributed version, add && env.rank == 0
                println("Will train until ", cb_early_stopping_eval, "'s ",
                        cb_early_stopping_metric[1], " hasn't improved in ",
                        early_stopping_rounds, " round(s).")
            end
        else
            if verbose_eval > 0 # For distributed version, add && env.rank == 0
                println("Will train until ",  cb_early_stopping_eval,
                        "'s default evaluation metric hasn't improved in ", early_stopping_rounds,
                        " round(s).")
            end
        end
    end

    function callback(env::CallbackEnv)
        cb_timing

        # If the early stopping metric isn't known yet, deduce it and reset scores.
        if cb_early_stopping_metric[1] == ""
            metric_names = keys(env.results[cb_early_stopping_eval])
            @assert length(metric_names) == 1
            early_stopping_metric = first(metric_names)
            cb_early_stopping_metric[1] = early_stopping_metric
            cb_best_iteration[1] = 0
            if should_maximize(early_stopping_metric)
                cb_maximize[1] = true
                cb_best_score[1] = -Inf
            else
                cb_maximize[1] = false
                cb_best_score[1] = Inf
            end
        end

        early_stopping_metric = cb_early_stopping_metric[1]
        maximize = cb_maximize[1]
        iteration = env.iteration
        best_score = cb_best_score[1]
        best_iteration = cb_best_iteration[1]

        # TODO: This only looks at the first column, will have to adapt this when used in CV.
        score = env.results[cb_early_stopping_eval][early_stopping_metric][iteration, 1]

        if (maximize && (score > best_score)) || (!maximize && (score < best_score))
            cb_best_score[1] = score
            cb_best_iteration[1] = iteration
            set_attr(env.model,
                     best_score = score,
                     best_iteration = iteration)
        elseif iteration - best_iteration >= early_stopping_rounds
            if verbose_eval
                println("Stopping. Best iteration: ", best_iteration, ", ", early_stopping_metric,
                        ": ", best_score)
            end
            shrink!(results, iteration)
            throw(EarlyStopException(best_iteration))
        end
    end

    return callback
end


# Remove all the rows in the matrices in the results after `last_retained_iter`.
function shrink!(results::Dict{String,Dict{String,Matrix{Float64}}}, last_retained_iter::Int)
    for eval_name in keys(results)
        curr_eval = results[eval_name]
        for metric_name in keys(test)
            curr_eval[metric_name] = curr_eval[metric_name][1:last_retained_iter, :]
        end
    end
    return nothing
end


# Return `true` if the metric should be maximized and `false` if it should be minimized.
function should_maximize(early_stopping_metric::String)
    if in(early_stopping_metric, maximize_metrics) ||
        any(i -> startswith(early_stopping_metric, i), maximize_at_n_metrics)
        return true
    end
    return false
end
