module XGBoost

include("XGBase.jl")

export xgboost, predict, save, nfold_cv, slice, Booster, DMatrix


### train ###
function xgboost(param::Dict{ASCIIString, Any}, dtrain::DMatrix, nrounds::Integer,
               watchlist::Array{DMatrix, 1}=[], obj=None, feval=None)
    plst = [(k, param[k]) for k in keys(param)]
    return xgboost(convert(Array{(ASCIIString, Any), 1}, plst), dtrain, watchlist, obj, feval)
end

function xgboost(param::Array{(ASCIIString, Any), 1}, dtrain::DMatrix, nrounds::Integer,
               watchlist::Array{DMatrix, 1}=[], obj=None, feval=None)
    push!(watchlist, dtrain)
    bst = Booster(watchlist, size(watchlist)[1])
    for itm in param
        set_param(bst, itm[1], string(itm[2])
    end
    for i = 1:nrounds
        update(bst, dtrain, obj, feval)
        XGBoosterEvalOneIter
    end
    return bst
end



function predict(bst::Booster, dmat::DMatrix,
                output_margin::Signed=0, ntree_limit::Integer=0)
    len = [1]
    ptr = XGBoosterPredict(bst.handle, dmat, convert(Int32, output_margin),
                            convert(Uint32, ntree_limit), length)
    return deepcopy(pointer_to_array(ptr, len[1]))
end


function nfold_cv()

end

### save ###
function save(bst::Booster, fname::ASCIIString)
    XGBoosterSaveModel(bst.handle, fname)
end

function save(dmat::DMatrix, fname::ASCIIString; slient::Signed=1)
    XGDMatrixSaveBinary(dmat.hanle, fname, convert(Int32, slient))
end

### dump model ###
function dump(bst::Booster, fmap::ASCIIString, out_len::Array{Integer, 1})
    XGBoosterDumpModel(bst.handle, fmap, convert(Array{Uint64, 1}, out_len))
end

### slice ###
function slice(dmat::DMatrix, idxset::Array{Signed, 1})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Array{Int32, 1}, idxset),
                                    size(idxset)[1])
    return DMatrix(handle)
end

end # module
