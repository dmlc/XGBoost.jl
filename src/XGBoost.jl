module XGBoost

include("XGBase.jl")

export xgboost, predict, save, nfold_cv, slice, Booster, DMatrix


### train ###
function xgboost(param::Dict{ASCIIString, Any}, dtrain::DMatrix, nrounds::Int64;
                 evals::Array{(DMatrix, ASCIIString), 1}=[], obj=None, feval=None)
    plst = (ASCIIString, ASCIIString)[]
    for k in keys(param)
        push!(plst, (k, string(param[k])))
    end
    return xgboost(plst, dtrain, nrounds, evals=evals, obj=obj, feval=feval)
end

function xgboost(param::Array{(ASCIIString, ASCIIString), 1},
                 dtrain::DMatrix, nrounds::Int64;
                 evals::Array{(DMatrix, ASCIIString), 1}=[], obj=None, feval=None)
    cache = [dtrain]
    for itm in evals
        push!(cache, itm[1])
    end
    bst = Booster(cache, size(cache)[1])
    for itm in param
        XGBoosterSetParam(bst.handle, itm[1], string(itm[2]))
    end
    dmats = DMatrix[]
    evnames = ASCIIString[]
    for itm in evals
        push!(dmats, itm[1])
        push!(evnames, itm[2])
    end
    for i = 1:nrounds
        if typeof(obj) == Function
            pred = predict(bst, dtrain)
            grad, hess = obj(pred, dtrain)
            @assert size(grad) == size(hess)
            XGBoosterBoostOneIter(bst.handle, convert(Int32, 1),
                                  convert(Array{Float32, 1}, grad),
                                  convert(Array{Float32, 1}, hess),
                                  convert(Uint64, size(hess)[1]))
        else
            XGBoosterUpdateOneIter(bst.handle, convert(Int32, 1), dtrain)
        end
        print(XGBoosterEvalOneIter(bst.handle, convert(Int32, i), dmats, evnames, convert(Uint64, size(dmats)[1])))
    end
    return bst
end



function predict(bst::Booster, dmat::DMatrix;
                 output_margin::Signed=0, ntree_limit::Integer=0)
    len = Uint64[1]
    ptr = XGBoosterPredict(bst.handle, dmat, convert(Int32, output_margin),
                            convert(Uint32, ntree_limit), len)
    return deepcopy(pointer_to_array(ptr, len[1]))
end


function nfold_cv()

end

### save ###
function save(bst::Booster, fname::ASCIIString)
    XGBoosterSaveModel(bst.handle, fname)
end

function save(dmat::DMatrix, fname::ASCIIString; slient::Signed=1)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, slient))
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

function get_label(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "label")
end

function get_weight(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "weight")
end

function get_margin(dmat::DMatrix)
    JLGetFloatInfo(dmat.handle, "base_margin")
end

end # module
