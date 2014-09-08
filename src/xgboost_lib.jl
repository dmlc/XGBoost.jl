include("xgboost_wrapper_h.jl")

# TODO: Use reference instead of array for length

type DMatrix
    handle::Ptr{Void}
    _set_info::Function
    function _setinfo{T<:Number}(ptr::Ptr{Void}, name::ASCIIString, array::Array{T, 1})
        if name == "label" || name == "weight" || name == "base_margin"
            XGDMatrixSetFloatInfo(ptr, name,
                                  convert(Array{Float32, 1}, array),
                                  convert(Uint64, size(array)[1]))
        elseif name == "group"
            XGDMatrixSetGroup(ptr, name,
                              convert(Array{Uint32, 1}, label),
                              convert(Uint64, size(array)[1]))
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
    function DMatrix{K<:Real, V<:Integer}(data::SparseMatrixCSC{K, V}; kwargs...)
        handle = XGDMatrixCreateFromCSC(convert(SparseMatrixCSC{Float32, Int64}, data))
        sp = new(handle,  _setinfo)
        for itm in kwargs
            _setinfo(handle, string(itm[1]), itm[2])
        end
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix{T<:Real}(data::Array{T, 2}, missing = 0;kwargs...)
        handle = XGDMatrixCreateFromMat(convert(Array{Float32, 2}, data),
                                        convert(Float32, missing))
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
                                   convert(Uint64, size(idxset)[1]))
    return DMatrix(handle)
end

type Booster
    handle::Ptr{Void}
    function Booster(dmats::Array{DMatrix, 1})
        handle = XGBoosterCreate([itm.handle for itm in dmats], size(dmats)[1])
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function Booster(fname::ASCIIString)
        handle = XGBoosterCreate(Ptr{Void}[], 0)
        XGBoosterLoadModel(handle, fname)
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
function dump_model(bst::Booster, fname::ASCIIString; fmap::ASCIIString="")
    out_len = Uint64[1]
    ptr = XGBoosterDumpModel(bst.handle, fmap, out_len)
    data = pointer_to_array(ptr, out_len[1])
    fo = open(fname, "w")
    for i=1:out_len[1]
        @printf(fo, "booster[%d]:\n", i)
        @printf(fo, "%s", bytestring(data[i]))
    end
    close(fo)
end

### train ###
function xgboost(dtrain::DMatrix, nrounds::Integer;
                 param=Any, watchlist=[],
                 obj=None, feval=None,
                 kwargs...)
    cache = [dtrain]
    for itm in watchlist
        push!(cache, itm[1])
    end
    bst = Booster(cache)
    for itm in kwargs
        print(itm, "\n")
        XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    end
    for itm in param
        XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    end

    for i = 1:nrounds
        update(bst, 1, dtrain, obj=obj)
        @printf(STDERR, "%s", eval_set(bst, watchlist, i, feval=feval))
    end
    return bst
end

### update ###
function update(bst::Booster, nrounds::Integer, dtrain::DMatrix; obj=None)
    if typeof(obj) == Function
        pred = predict(bst, dtrain)
        grad, hess = obj(pred, dtrain)
        @assert size(grad) == size(hess)
        XGBoosterBoostOneIter(bst.handle, dtrain.handle,
                              convert(Array{Float32, 1}, grad),
                              convert(Array{Float32, 1}, hess),
                              convert(Uint64, size(hess)[1]))
    else
        XGBoosterUpdateOneIter(bst.handle, convert(Int32, nrounds), dtrain.handle)
    end
end


### eval_set ###
function eval_set(bst::Booster, watchlist::Array{(DMatrix, ASCIIString), 1},
                  iter::Integer; feval=None)
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
                                                     evnames, convert(Uint64, size(dmats)[1])))
    end
    return res
end

### predict ###
function predict(bst::Booster, dmat::DMatrix;
                 output_margin::Bool = false, ntree_limit::Integer=0)
    len = Uint64[1]
    ptr = XGBoosterPredict(bst.handle, dmat.handle, convert(Int32, output_margin),
                           convert(Uint32, ntree_limit), len)
    return deepcopy(pointer_to_array(ptr, len[1]))
end

type CVPack
    dtrain::DMatrix
    dtest::DMatrix
    watchlist::Array{(DMatrix, ASCIIString), 1}
    bst::Booster
    function CVPack(dtrain::DMatrix, dtest::DMatrix, param)
        bst = Booster([dtrain,dtest])
        for itm in param
            XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
        end
        watchlist = [ (dtrain,"train"), (dtest, "test") ]
        new(dtrain, dtest, watchlist, bst)
    end
end

function mknfold(dall::DMatrix, nfold::Integer, param,
                 seed::Integer, evals=[]; fpreproc = None)
    srand(seed)
    randidx = randperm(XGDMatrixNumRow(dall.handle))
    kstep = int(size(randidx)[1] / nfold)
    idset = [randidx[ ((i - 1)*kstep) + 1 : min(size(randidx)[1],(i)*kstep + 1) ] for i=1:nfold]
    ret = CVPack[]
    for k=1:nfold
        selected = []
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
            if k in keys(cvmap) == false
                cvmap[k] = Float64[]
            end
            push!(cvmap[k], float(v))
        end
    end
    itms = [itm for itm in cvmap]
    sort!(itms)
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

function nfold_cv(params, dtrain::DMatrix, num_boost_round::Integer=10,
                  nfold::Integer=3; metrics=[], obj = None, feval = None,
                  fpreproc = None, show_stdv=true, seed::Integer=0)
    results = ASCIIString[]
    cvfolds = mknfold(dtrain, nfold, params, seed, metrics, fpreproc=fpreproc)
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

