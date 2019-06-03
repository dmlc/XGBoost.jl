# TODO: Use reference instead of array for length

mutable struct DMatrix
    handle::Ptr{Nothing}
    _set_info::Function

    function _setinfo(ptr::Ptr{Nothing}, name::String, array::Vector{<: Number})
        if name == "label" || name == "weight" || name == "base_margin"
            XGDMatrixSetFloatInfo(ptr, name,
                                  convert(Vector{Float32}, array),
                                  convert(UInt64, size(array)[1]))
        elseif name == "group"
            XGDMatrixSetGroup(ptr,
                              convert(Vector{UInt32}, array),
                              convert(UInt64, size(array)[1]))
        else
            error("unknown information name")
        end
    end

    function DMatrix(handle::Ptr{Nothing})
        dmat = new(handle, _setinfo)
        finalizer(JLFree, dmat)
        return dmat
    end

    function DMatrix(fname::String; silent = false)
        handle = XGDMatrixCreateFromFile(fname, convert(Int32, silent))
        dmat = new(handle, _setinfo)
        finalizer(JLFree, dmat)
        return dmat
    end

    function DMatrix(data::SparseMatrixCSC{K,V}, transposed::Bool = false;
                                          kwargs...) where {K<:Real, V<:Integer}
        handle = (transposed ? XGDMatrixCreateFromCSCT(data) : XGDMatrixCreateFromCSC(data))
        for itm in kwargs
            _setinfo(handle, string(itm[1]), itm[2])
        end
        dmat = new(handle, _setinfo)
        finalizer(JLFree, dmat)
        return dmat
    end

    function DMatrix(data::Matrix{<:Real}, transposed::Bool = false, missing = NaN32;
                              kwargs...)
        handle = nothing
        if !transposed
            handle = XGDMatrixCreateFromMat(convert(Matrix{Float32}, data),
                                            convert(Float32, missing))
        else
            handle = XGDMatrixCreateFromMatT(convert(Matrix{Float32}, data),
                                             convert(Float32, missing))
        end

        for itm in kwargs
            _setinfo(handle, string(itm[1]), itm[2])
        end
        dmat = new(handle, _setinfo)
        finalizer(JLFree, dmat)
        return dmat
    end

    function JLFree(dmat::DMatrix)
        XGDMatrixFree(dmat.handle)
    end
end

function get_info(dmat::DMatrix, field::String)
    JLGetFloatInfo(dmat.handle, field)
end

function set_info(dmat::DMatrix, field::String, array::Vector{<:Real})
    dmat._set_info(dmat.handle, field, array)
end

function save(dmat::DMatrix, fname::String; silent = true)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, silent))
end

### slice ###
function slice(dmat::DMatrix, idxset::Vector{<:Real})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Vector{Int32}, idxset .- 1),
                                   convert(UInt64, size(idxset)[1]))
    return DMatrix(handle)
end

mutable struct Booster
    handle::Ptr{Nothing}

    function Booster(; cachelist::Vector{DMatrix} = convert(Vector{DMatrix}, []),
                     model_file::String = "")
        handle = XGBoosterCreate([itm.handle for itm in cachelist], size(cachelist)[1])
        if model_file != ""
            XGBoosterLoadModel(handle, model_file)
        end
        bst = new(handle)
        finalizer(JLFree, bst)
        return bst
    end

    function JLFree(bst::Booster)
        XGBoosterFree(bst.handle)
    end
end

### save ###
function save(bst::Booster, fname::String)
    XGBoosterSaveModel(bst.handle, fname)
end

### dump model ###
function dump_model(bst::Booster, fname::String; fmap::String="", with_stats::Bool = false)
    data = XGBoosterDumpModel(bst.handle, fmap, convert(Int64, with_stats))
    fo = open(fname, "w")
    for i in 1:length(data)
        @printf(fo, "booster[%d]:\n", i)
        @printf(fo, "%s", unsafe_string(data[i]))
    end
    close(fo)
end

function makeDMatrix(data, label)
    # running converts
    if typeof(data) != DMatrix
        if typeof(data) == String
            if label != Union{}
                warning("label will be ignored when data is a file")
            end
            return DMatrix(data)
        else
            if label == Union{}
                error("label argument must be present for training, unless you pass in a DMatrix")
            end
            return DMatrix(data, label = label)
        end
    else
        return data
    end
end

### train ###
function xgboost(data, nrounds::Integer; label = Union{}, param = [], watchlist = [], metrics = [],
                 obj = Union{}, feval = Union{}, group = [], kwargs...)
    dtrain = makeDMatrix(data, label)
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
    for i = 1:nrounds
        update(bst, 1, dtrain, obj=obj)
        if !silent
            @printf(stderr, "%s", eval_set(bst, watchlist, i, feval = feval))
        end
    end
    return bst
end

### update ###
function update(bst::Booster, nrounds::Integer, dtrain::DMatrix; obj = Union{})
    if isa(obj, Function)
        pred = predict(bst, dtrain)
        grad, hess = obj(pred, dtrain)
        @assert size(grad) == size(hess)
        XGBoosterBoostOneIter(bst.handle, dtrain.handle,
                              convert(Vector{Float32}, grad),
                              convert(Vector{Float32}, hess),
                              convert(UInt64, size(hess)[1]))
    else
        XGBoosterUpdateOneIter(bst.handle, convert(Int32, nrounds), dtrain.handle)
    end
end


### eval_set ###
function eval_set(bst::Booster, watchlist::Vector{Tuple{DMatrix,String}}, iter::Integer;
                  feval = Union{})
    dmats = DMatrix[]
    evnames = String[]
    for itm in watchlist
        push!(dmats, itm[1])
        push!(evnames, itm[2])
    end
    res = ""
    if isa(feval, Function)
        res *= @sprintf("[%d]", iter)
        #@printf(stderr, "[%d]", iter)
        for j in 1:size(dmats)[1]
            pred = predict(bst, dmats[j])
            name, val = feval(pred, dmats[j])
            res *= @sprintf("\t%s-%s:%f", evnames[j], name, val)
        end
        res *= @sprintf("\n")
    else
        res *= @sprintf("%s\n", XGBoosterEvalOneIter(bst.handle, convert(Int32, iter),
                                                     [mt.handle for mt in dmats],
                                                     evnames, convert(UInt64, size(dmats)[1])))
    end
    return res
end

### predict ###
function predict(bst::Booster, data; output_margin::Bool = false, ntree_limit::Integer = 0)
    if typeof(data) != DMatrix
        data = DMatrix(data)
    end

    len = UInt64[1]
    ptr = XGBoosterPredict(bst.handle, data.handle, convert(Int32, output_margin),
                           convert(UInt32, ntree_limit), len)
    return deepcopy(unsafe_wrap(Array, ptr, len[1]))
end

mutable struct CVPack
    dtrain::DMatrix
    dtest::DMatrix
    watchlist::Vector{Tuple{DMatrix,String}}
    bst::Booster
    function CVPack(dtrain::DMatrix, dtest::DMatrix, param)
        bst = Booster(cachelist = [dtrain, dtest])
        for itm in param
            XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
        end
        watchlist = [(dtrain,"train"), (dtest, "test")]
        new(dtrain, dtest, watchlist, bst)
    end
end

function mknfold(dall::DMatrix, nfold::Integer, param, seed::Integer, evals=[]; fpreproc = Union{},
                 kwargs = [])
    seed!(seed)
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
            push!(cvmap[k], parse(Float64, v))
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

function nfold_cv(data, num_boost_round::Integer = 10, nfold::Integer = 3; label = Union{},
                  param=[], metrics=[], obj = Union{}, feval = Union{}, fpreproc = Union{},
                  show_stdv = true, seed::Integer = 0, kwargs...)
    dtrain = makeDMatrix(data, label)
    results = String[]
    cvfolds = mknfold(dtrain, nfold, param, seed, metrics, fpreproc=fpreproc, kwargs = kwargs)
    for i in 1:num_boost_round
        for f in cvfolds
            update(f.bst, 1, f.dtrain, obj = obj)
        end
        res = aggcv([eval_set(f.bst, f.watchlist, i, feval = feval) for f in cvfolds],
                    show_stdv = show_stdv)
        push!(results, res)
        @printf(stderr, "%s\n", res)
    end
end

struct FeatureImportance
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
            if typeof(m) != Nothing
                fname = replace(m.captures[1], nameStrip => "")

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
