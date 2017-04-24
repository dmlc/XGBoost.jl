# function train(bst::Booster, params::Dict, dtrain::DMatrix;
#                num_boost_round::Int = 10,
#                evals::Vector{Tuple{DMatrix,String}} = Vector{Tuple{DMatrix,String}}(),
#                obj::Union{Function,Void} = nothing, feval::Union{Function,Void} = nothing,
#                maximize::Bool = false, early_stopping_rounds::Union{Int,Void} = nothing,
#                evals_result::Union{Dict{String,Dict{String,String}},Void} = nothing,
#                verbose_eval::Union{Bool,Int} = true, xgb_model::Union{String,Void} = nothing,
#                callbacks::Union{Vector{Function},Void} = nothing)

#     callbacks_vec = typeof(callbacks) == Vector{Function} ? callbacks : Vector{Function}()

#     if typeof(verbose_eval) == Bool && verbose_eval == true
#         push!(callbacks_vec, cb_print_evaluation()) # TODO: Add this callback
#     elseif typeof(verbose_eval) == Int
#         push!(callbacks_vec, cb_print_evaluation(verbose_eval))
#     end

#     if typeof(early_stopping_rounds) != Void
#         push!(callbacks_vec, cb_early_stop(early_stopping_rounds; maximize = maximize,
#                                            verbose = verbose_eval))
#     end

#     if typeof(evals_result) != Void
#         push!(callbacks_vec, cb_record_evaluation(evals_result))
#     end

#     if typeof(xgb_model) != Void
#         bst = Booster(params, unshift!([eval[1] for eval in evals], dtrain); model_file = xgb_model)
#         num_boost = length(get_dump(bst))
#     else
#         bst = Booster(params, unshift!([eval[1] for eval in evals], dtrain))
#         num_boost = 0
#     end


#     # Add support for parallel tree
#     num_parallel_tree = 1
#     # Add support for num_class
#     # Add distributed code support
#     start_iteration = 1

#     callbacks_before_iter = [cb for cb in callbacks_vec if cb("before")]
#     callbacks_after_iter = [cb for cb in callbacks_vec if cb("after")]

#     for i in start_iteration:num_boost_round
#         # for cb in callbacks_before_iter
#         #     cb(CallbackEnv(bst, CVPack[], i, start_iteration, num_boost_round, rank,
#         #                    Dict{String,Matrix{Float64}}())
#         # end

#         # Add distributed code support

#         num_boost += 1

#         evaluation_result_list = []
#         if length(evals) > 0
#             bst_eval_set = eval_set(bst, evals, i, feval)
#             if typeof(bst_eval_set) == String
#                 msg = bst_eval_set
#             else
#                 msg = decode(bst_eval_set) # TODO: implement decode
#             end
#             res = [split(x, ':') for x in split(msg)]
#             evaluation_result_list = [(k, float(v)) for (k, v) in res] # TODO: res starts at idx 2?
#         end

#         try
#             for cb in callbacks_after_iter
#                 cb(callbackenv)
#             end
#         catch err
#             if typeof(err) == EarlyStopException
#                 break
#             else
#                 throw(err)
#             end
#         end
#         # Add distributed code
#     end

#     if typeof(attr(bst, "best_score")) != Void
#         bst.best_score = float(attr(bst, "best_score"))
#         bst.best_iteration = parse(Int, attr(bst, "best_iteration"))
#     else
#         bst.best_iteration = num_boost - 1
#     end
#     bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree

#     return bst
# end

### train ###
function xgboost(data, nrounds::Integer;
                 label = nothing, param::Dict{String,<:Any} = Dict{String,String}(),
                 watchlist = [], metrics = [],
                 obj = nothing, feval = nothing, group = [], kwargs...)
    if typeof(data) != DMatrix
        dtrain = DMatrix(data, label = label)
    else
        dtrain = data
    end

    if length(group) > 0
      set_info(dtrain, "group", group)
    end

    cache = [dtrain]
    for itm in watchlist
        push!(cache, itm[1])
    end
    bst = Booster(cachelist = cache)
    XGBoosterSetParam(bst.handle, "silent", "1")
    silent = false
    for itm in kwargs
        XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
        if itm[1] == :silent
            silent = itm[2] != 0
        end
    end
    for itm in param
        XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    end
    if size(watchlist)[1] == 0
        watchlist = [(dtrain, "train")]
    end
    for itm in metrics
        XGBoosterSetParam(bst.handle, "eval_metric", string(itm))
    end
    for i in 1:nrounds
        update(bst, dtrain, i, fobj = obj)
        if !silent
            @printf(STDERR, "%s", eval_set(bst, watchlist, i, feval = feval))
        end
    end
    return bst
end


type CVPack
    dtrain::DMatrix
    dtest::DMatrix
    watchlist::Vector{Tuple{DMatrix,String}}
    bst::Booster

    function CVPack(dtrain::DMatrix, dtest::DMatrix, param)
        bst = Booster(cachelist = [dtrain, dtest])
        for itm in param
            XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
        end
        watchlist = [(dtrain, "train"), (dtest, "test")]
        new(dtrain, dtest, watchlist, bst)
    end
end


function mknfold(dall::DMatrix, nfold::Integer, param, seed::Integer, evals=[]; fpreproc = nothing,
                 kwargs = [])
    srand(seed)
    randidx = randperm(XGDMatrixNumRow(dall.handle))
    kstep = size(randidx)[1] / nfold
    idset = [randidx[round(Int64, (i-1) * kstep) + 1 : min(size(randidx)[1],round(Int64, i * kstep))] for i in 1:nfold]
    ret = CVPack[]
    for k in 1:nfold
        selected = Int[]
        for i in 1:nfold
            if k != i
                selected = vcat(selected, idset[i])
            end
        end
        dtrain = slice(dall, selected)
        dtest = slice(dall, idset[k])
        if typeof(fpreproc) == Function
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, deepcopy(param))
        else
            tparam = param
        end
        plst = vcat([itm for itm in param], [("eval_metric", itm) for itm in evals])
        plst = vcat(plst, [(string(itm[1]), string(itm[2])) for itm in kwargs])
        push!(ret, CVPack(dtrain, dtest, plst))
    end
    return ret
end


function aggcv(rlist; show_stdv = true)
    cvmap = Dict()
    ret = split(rlist[1])[1]
    for line in rlist
        arr = split(line)
        @assert ret == arr[1]
        for it in arr[2:end]
            k, v  = split(it, ":")
            if !haskey(cvmap, k)
                cvmap[k] = Float64[]
            end
            push!(cvmap[k], float(v))
        end
    end
    items = [item for item in cvmap]
    sort!(items, by = x -> x[1])
    for item in items
        k = item[1]
        v = item[2]
        if show_stdv == true
            ret *= @sprintf("\tcv-%s:%f+%f", k, mean(v), std(v))
        else
            ret *= @sprintf("\tcv-%s:%f", k, mean(v))
        end
    end
    return ret
end


function nfold_cv(data, num_boost_round::Integer = 10, nfold::Integer = 3; label = nothing,
                  param = [], metrics = [], obj = nothing, feval = nothing, fpreproc = nothing,
                  show_stdv = true, seed::Integer = 0, kwargs...)
    if typeof(data) != DMatrix
        dtrain = DMatrix(data, label = label)
    else
        dtrain = data
    end
    results = String[]
    cvfolds = mknfold(dtrain, nfold, param, seed, metrics, fpreproc = fpreproc, kwargs = kwargs)
    for i in 1:num_boost_round
        for f in cvfolds
            update(f.bst, f.dtrain, i, fobj = obj)
        end
        res = aggcv([eval_set(f.bst, f.watchlist, i, feval = feval) for f in cvfolds],
                    show_stdv = show_stdv)
        push!(results, res)
        @printf(STDERR, "%s\n", res)
    end
end


immutable FeatureImportance
    fname::String
    gain::Float64
    cover::Float64
    freq::Float64
end


function Base.show(io::IO, f::FeatureImportance)
    @printf(io, "%s: gain = %0.04f, cover = %0.04f, freq = %0.04f", f.fname, f.gain, f.cover,
            f.freq)
end


function Base.show(io::IO, arr::Vector{FeatureImportance}; maxrows = 30)
    println(io, "$(length(arr))-element Vector{$(FeatureImportance)}:")
    println(io, "Gain      Coverage  Frequency  Feature")
    for i in 1:min(maxrows, length(arr))
        @printf(io, "%0.04f    %0.04f    %0.04f     %s\n", arr[i].gain, arr[i].cover, arr[i].freq,
                arr[i].fname)
    end
end


Base.show(io::IO, ::MIME"text/plain", arr::Vector{FeatureImportance}) = show(io, arr)


function importance(bst::Booster; fmap::String = "")
    data = XGBoosterDumpModel(bst.handle, fmap, 1)

    # get the total gains for each feature and the whole model
    gains = Dict{String,Float64}()
    covers = Dict{String,Float64}()
    freqs = Dict{String,Float64}()
    totalGain = 0.
    totalCover = 0.
    totalFreq = 0.
    lineMatch = r"^[^\w]*[0-9]+:\[([^\]]+)\] yes=([\.+e0-9]+),no=([\.+e0-9]+),[^,]*,?gain=([\.+e0-9]+),cover=([\.+e0-9]+).*"
    nameStrip = r"[<>][^<>]+$"
    for i in 1:length(data)
        for line in split(unsafe_string(data[i]), '\n')
            m = match(lineMatch, line)
            if typeof(m) != Void
                fname = replace(m.captures[1], nameStrip, "")

                gain = parse(Float64, m.captures[4])
                totalGain += gain
                gains[fname] = get(gains, fname, 0.) + gain

                cover = parse(Float64, m.captures[5])
                totalCover += cover
                covers[fname] = get(covers, fname, 0.) + cover

                totalFreq += 1
                freqs[fname] = get(freqs, fname, 0.) + 1
            end
        end
    end

    # compile these gains into list of features sorted by gain value
    res = FeatureImportance[]
    for fname in keys(gains)
        push!(res, FeatureImportance(fname,
                                     gains[fname] / totalGain,
                                     covers[fname] / totalCover,
                                     freqs[fname] / totalFreq))
    end
    sort!(res, by = x -> -x.gain)
end


function importance(bst::Booster, feature_names::Vector{String})
    res = importance(bst)

    result = FeatureImportance[]
    for old_importance in res
        actual_name = feature_names[parse(Int64, old_importance.fname[2:end]) + 1]
        push!(result, FeatureImportance(actual_name, old_importance.gain, old_importance.cover,
                                        old_importance.freq))
    end

    return result
end
