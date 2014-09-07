include("xgboost_wrapper_h.jl")

type DMatrix
    handle::Ptr{Void}
    function _setinfo(ptr::Ptr{Void}, name::ASCIIString, array::Array{Any,1})
        if name == "label" || name == "weight" || name == "base_margin"
            XGDMatrixSetFloatInfo(ptr, name,
                                  convert(Array{Float32, 1}, array),
                                  size(array)[1])
        elseif name == "group"
            XGDMatrixSetGroup(ptr, name,
                              convert(Array{Uint32, 1}, label),
                              size(array)[1])
        else
            error("unknown information name") 
        end
    end
    function DMatrix(handle::Ptr{Void})
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(fname::ASCIIString; silent = 0)
        handle = XGDMatrixCreateFromFile(fname, convert(Int32, silent))
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(data::SparseMatrixCSC{Float32, Int64}; kwargs...)
        handle = XGDMatrixCreateFromCSC(data)
        sp = new(handle)
        for itm in kwargs
            _setinfo(handle, itm[1], itm[2])
        end
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(data::Array{Float32, 2}; missing::Float32=0,
                     kwargs...)
        handle = XGDMatrixCreateFromMat(data)
        for itm in kwargs
            _setinfo(handle, itm[1], itm[2])
        end
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function JLFree(dmat::DMatrix)
        XGDMatrixFree(dmat.handle)
    end    
end

function get_label(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "label")
end

function get_weight(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "weight")
end

function get_margin(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "base_margin")
end

function save(dmat::DMatrix, fname::ASCIIString; slient::Signed=1)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, slient))
end

### slice ###
function slice(dmat::DMatrix, idxset::Array{Signed, 1})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Array{Int32, 1}, idxset),
                                   size(idxset)[1])
    return DMatrix(handle)
end

type Booster
    handle::Ptr{Void}
    function Booster(dmats::Array{DMatrix, 1}, len::Int64)
        handle = XGBoosterCreate([itm.handle for itm in dmats], len::Int64)
        new(handle)
        sp = new(handle)        
        finalizer(sp, JLFree)
        sp
    end
    function Booster(fname::ASCIIString)
        handle = XGBoosterCreate(Ptr{Void}[], 0)
        XGBoosterLoadModel(handle, fname)
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
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
function dump(bst::Booster, fmap::ASCIIString, out_len::Array{Integer, 1})
    XGBoosterDumpModel(bst.handle, fmap, convert(Array{Uint64, 1}, out_len))
end


### train ###
function xgboost(dtrain::DMatrix, nrounds::Integer;
                 param=[], watchlist=[],
                 obj=None, feval=None,
                 kwargs...)
    cache = [dtrain]
    for itm in watchlist
        push!(cache, itm[1])
    end
    bst = Booster(cache, size(cache)[1])
    for itm in kwargs
        XGBoosterSetParam(bst.handle, string(itm[1]), string(itm[2]))
    end
    dmats = DMatrix[]
    evnames = ASCIIString[]
    for itm in watchlist
        push!(dmats, itm[1])
        push!(evnames, itm[2])
    end
    for i = 1:nrounds
        if typeof(obj) == Function
            pred = predict(bst.handle, dtrain)
            grad, hess = obj(pred, dtrain)
            @assert size(grad) == size(hess)
            XGBoosterBoostOneIter(bst.handle, convert(Int32, 1),
                                  convert(Array{Float32, 1}, grad),
                                  convert(Array{Float32, 1}, hess),
                                  convert(Uint64, size(hess)[1]))
        else
            XGBoosterUpdateOneIter(bst.handle, convert(Int32, 1), dtrain.handle)
        end
        print(XGBoosterEvalOneIter(bst.handle, convert(Int32, i), [mt.handle for mt in dmats],
                                   evnames, convert(Uint64, size(dmats)[1])))
        print("\n")
    end
    return bst
end

function predict(bst::Booster, dmat::DMatrix;
                 output_margin::Bool = false, ntree_limit::Integer=0)
    len = Uint64[1]
    ptr = XGBoosterPredict(bst.handle, dmat.handle, convert(Int32, output_margin),
                           convert(Uint32, ntree_limit), len)
    return deepcopy(pointer_to_array(ptr, len[1]))
end


function nfold_cv()

end

