using Compat

include("xgboost_wrapper_h.jl")

# TODO: Use reference instead of array for length

type DMatrix
    handle::Ptr{Void}
    _set_info::Function
    function _setinfo{T<:Number}(ptr::Ptr{Void}, name::ASCIIString, array::Array{T, 1})
        if name == "label" || name == "weight" || name == "base_margin"
            XGDMatrixSetFloatInfo(ptr, name,
                                  convert(Array{Float32, 1}, array),
                                  convert(UInt64, size(array)[1]))
        elseif name == "group"
            XGDMatrixSetGroup(ptr,
                              convert(Array{UInt32, 1}, array),
                              convert(UInt64, size(array)[1]))
        else
            error("unknown information name")
        end
    end
    function DMatrix(handle::Ptr{Void})
        sp = new(handle, _setinfo)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(fname::ASCIIString; silent = false)
        handle = XGDMatrixCreateFromFile(fname, convert(Int32, silent))
        sp = new(handle, _setinfo)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix{K<:Real, V<:Integer}(data::SparseMatrixCSC{K, V}, transposed::Bool=false; kwargs...)
        handle = (transposed ? XGDMatrixCreateFromCSCT(data) : XGDMatrixCreateFromCSC(data))
        for itm in kwargs
            _setinfo(handle, string(itm[1]), itm[2])
        end
        sp = new(handle, _setinfo)
        finalizer(sp, JLFree)
        sp
    end

    function DMatrix{T<:Real}(data::Array{T, 2}, transposed::Bool=false, missing = NaN32; kwargs...)
        handle = nothing
        if !transposed
            handle = XGDMatrixCreateFromMat(convert(Array{Float32, 2}, data), convert(Float32, missing))
        else
            handle = XGDMatrixCreateFromMatT(convert(Array{Float32, 2}, data), convert(Float32, missing))
        end

        for itm in kwargs
            _setinfo(handle, string(itm[1]), itm[2])
        end
        sp = new(handle, _setinfo)
        finalizer(sp, JLFree)
        sp
    end
    function JLFree(dmat::DMatrix)
        XGDMatrixFree(dmat.handle)
    end
end

function get_info(dmat::DMatrix, field::ASCIIString)
    JLGetFloatInfo(dmat.handle, field)
end

function set_info{T<:Real}(dmat::DMatrix, field::ASCIIString, array::Array{T, 1})
    dmat._set_info(dmat.handle, field, array)
end

function save(dmat::DMatrix, fname::ASCIIString; slient=true)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, slient))
end

### slice ###
function slice{T<:Integer}(dmat::DMatrix, idxset::Array{T, 1})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Array{Int32, 1}, idxset) - 1,
                                   convert(UInt64, size(idxset)[1]))
    return DMatrix(handle)
end

type Booster
    handle::Ptr{Void}
    function Booster(;cachelist::Array{DMatrix, 1} = convert(Array{DMatrix,1}, []),
                     model_file::ASCIIString = "")
        handle = XGBoosterCreate([itm.handle for itm in cachelist], size(cachelist)[1])
        if model_file != ""
            XGBoosterLoadModel(handle, model_file)
        end
        this = new(handle)
        finalizer(this, JLFree)
        return this
    end
    function JLFree(bst::Booster)
        XGBoosterFree(bst.handle)
    end
end

### save ###
function save(bst::Booster, fname::ASCIIString)
    XGBoosterSaveModel(bst.handle, fname)
end

### dump model ###
function dump_model(bst::Booster, fname::ASCIIString; fmap::ASCIIString="", with_stats::Bool=false)
    data = XGBoosterDumpModel(bst.handle, fmap, convert(Int64, with_stats))
    fo = open(fname, "w")
    for i=1:length(data)
        @printf(fo, "booster[%d]:\n", i)
        @printf(fo, "%s", bytestring(data[i]))
    end
    close(fo)
end

function makeDMatrix(data, label)
    # running converts
    if typeof(data) != DMatrix
        if typeof(data) == ASCIIString
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
function xgboost(data, nrounds::Integer;
                 label=Union{}, param=[], watchlist=[], metrics=[],
                 obj=Union{}, feval=Union{}, group=[],
                 kwargs...)
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
            @printf(STDERR, "%s", eval_set(bst, watchlist, i, feval=feval))
        end
    end
    return bst
end

### update ###
function update(bst::Booster, nrounds::Integer, dtrain::DMatrix; obj=Union{})
    if typeof(obj) == Function
        pred = predict(bst, dtrain)
        grad, hess = obj(pred, dtrain)
        @assert size(grad) == size(hess)
        XGBoosterBoostOneIter(bst.handle, dtrain.handle,
                              convert(Array{Float32, 1}, grad),
                              convert(Array{Float32, 1}, hess),
                              convert(UInt64, size(hess)[1]))
    else
        XGBoosterUpdateOneIter(bst.handle, convert(Int32, nrounds), dtrain.handle)
    end
end


### eval_set ###
function eval_set(bst::Booster, watchlist::Array{(@compat Tuple{DMatrix, ASCIIString}), 1},
                  iter::Integer; feval=Union{})
    dmats = DMatrix[]
    evnames = ASCIIString[]
    for itm in watchlist
        push!(dmats, itm[1])
        push!(evnames, itm[2])
    end
    res = ""
    if typeof(feval) == Function
        res *= @sprintf("[%d]", iter)
        #@printf(STDERR, "[%d]", iter)
        for j=1:size(dmats)[1]
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
function predict(bst::Booster, data;
                 output_margin::Bool = false, ntree_limit::Integer=0)
    if typeof(data) != DMatrix
        data = DMatrix(data)
    end

    len = UInt64[1]
    ptr = XGBoosterPredict(bst.handle, data.handle, convert(Int32, output_margin),
                           convert(UInt32, ntree_limit), len)
    return deepcopy(pointer_to_array(ptr, len[1]))
end

type CVPack
    dtrain::DMatrix
    dtest::DMatrix
    watchlist::Array{(@compat Tuple{DMatrix, ASCIIString}), 1}
    bst::Booster
    function CVPack(dtrain::DMatrix, dtest::DMatrix, param)
        bst = Booster(cachelist = [dtrain,dtest])
        for itm in param
            XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
        end
        watchlist = [ (dtrain,"train"), (dtest, "test") ]
        new(dtrain, dtest, watchlist, bst)
    end
end

function mknfold(dall::DMatrix, nfold::Integer, param,
                 seed::Integer, evals=[]; fpreproc = Union{}, kwargs = [])
    srand(seed)
    randidx = randperm(XGDMatrixNumRow(dall.handle))
    kstep = size(randidx)[1] / nfold
    idset = [randidx[ round(Int64, (i-1)*kstep) + 1 : min(size(randidx)[1],round(Int64, i*kstep)) ] for i=1:nfold]
    ret = CVPack[]
    for k=1:nfold
        selected = Array(Int,0)
        for i=1:nfold
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

function aggcv(rlist; show_stdv=true)
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
    itms = [itm for itm in cvmap]
    sort!(itms, by=x->x[1])
    for itm in itms
        k = itm[1]
        v = itm[2]
        if show_stdv == true
            ret *= @sprintf("\tcv-%s:%f+%f", k, mean(v), std(v))
        else
            ret *= @sprintf("\tcv-%s:%f", k, mean(v))
        end
    end
    return ret
end

function nfold_cv(data, num_boost_round::Integer=10, nfold::Integer=3;
                  label = Union{}, param=[], metrics=[], obj = Union{}, feval = Union{},
                  fpreproc = Union{}, show_stdv=true, seed::Integer=0, kwargs...)
    dtrain = makeDMatrix(data, label)
    results = ASCIIString[]
    cvfolds = mknfold(dtrain, nfold, param, seed, metrics, fpreproc=fpreproc, kwargs = kwargs)
    for i=1:num_boost_round
        for f in cvfolds
            update(f.bst, 1, f.dtrain, obj=obj)
        end
        res = aggcv([eval_set(f.bst, f.watchlist, i, feval=feval) for f in cvfolds],
                    show_stdv=show_stdv)
        push!(results, res)
        @printf(STDERR, "%s\n", res)
    end
end

immutable FeatureImportance
    fname::ASCIIString
    gain::Float64
    cover::Float64
    freq::Float64
end

function Base.show(io::IO, f::FeatureImportance)
    @printf(io, "%s: gain = %0.04f, cover = %0.04f, freq = %0.04f", f.fname, f.gain, f.cover, f.freq)
end

function Base.show(io::IO, arr::Array{FeatureImportance,1}; maxrows=30)
    println(io, "$(length(arr))-element Array{$(FeatureImportance),1}:")
    println(io, "Gain      Coverage  Frequency  Feature")
    for i in 1:min(maxrows, length(arr))
        @printf(io, "%0.04f    %0.04f    %0.04f     %s\n", arr[i].gain, arr[i].cover, arr[i].freq, arr[i].fname)
    end
end

Base.writemime(io::IO, ::MIME"text/plain", arr::Array{FeatureImportance,1}) = show(io, arr)

function importance(bst::Booster; fmap::ASCIIString="")
    data = XGBoosterDumpModel(bst.handle, fmap, 1)

    # get the total gains for each feature and the whole model
    gains = Dict{ASCIIString,Float64}()
    covers = Dict{ASCIIString,Float64}()
    freqs = Dict{ASCIIString,Float64}()
    totalGain = 0.0
    totalCover = 0.0
    totalFreq = 0.0
    lineMatch = r"^[^\w]*[0-9]+:\[([^\]]+)\] yes=([\.+e0-9]+),no=([\.+e0-9]+),[^,]*,?gain=([\.+e0-9]+),cover=([\.+e0-9]+).*"
    nameStrip = r"[<>][^<>]+$"
    for i=1:length(data)
        for line in split(bytestring(data[i]), '\n')
            m = match(lineMatch, line)
            if typeof(m) != Void
                fname = replace(m.captures[1], nameStrip, "")

                gain = parse(Float64, m.captures[4])
                totalGain += gain
                gains[fname] = get(gains, fname, 0.0) + gain

                cover = parse(Float64, m.captures[5])
                totalCover += cover
                covers[fname] = get(covers, fname, 0.0) + cover

                totalFreq += 1
                freqs[fname] = get(freqs, fname, 0.0) + 1
            end
        end
    end

    # compile these gains into list of features sorted by gain value
    res = FeatureImportance[]
    for fname in keys(gains)
        push!(res, FeatureImportance(
            fname,
            gains[fname]/totalGain,
            covers[fname]/totalCover,
            freqs[fname]/totalFreq
        ))
    end
    sort!(res, by=x->-x.gain)
end

function importance(bst::Booster, feature_names::Vector{ASCIIString})
    res = importance(bst)

    result = FeatureImportance[]
    for old_importance in res
        actual_name = feature_names[parse(Int64, old_importance.fname[2:end])+1]
        push!(result, FeatureImportance(actual_name, old_importance.gain,
                                    old_importance.cover, old_importance.freq))
    end

    result
end
