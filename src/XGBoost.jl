module XGBoost

include("XGBase.jl")

### train ###
function train(param::Dict{ASCIIString, Any}, dtrain::DMatrix, nrounds::Integer,
               watchlist::Array{DMatrix, 1}=[], obj=None, feval=None)
    plst = [(k, param[k]) for k in keys(param)]
    return train(convert(Array{(ASCIIString, Any), 1}, plst), dtrain, watchlist, obj, feval)

end

function train(param::Array{(ASCIIString, Any), 1}, dtrain::DMatrix, nrounds::Integer,
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


function update(bst::Booster, nrounds::Integer, dtrain::DMatrix, obj=None)
    if typeof(obj) == typeof(update)
        grad, hess = obj(dtrain)
        @assert size(grad) == size(hess)
        XGBoosterUpdateOneIter(bst, convert(Int32, nrounds), dtrain, 
                               convert(Array{Float32, 1}, grad),
                               convert(Array{Float32, 1}, hess),
                               size(grad)[1])
    else
        XGBoosterUpdateOneIter(bst, convert(Int32, nrounds), dtrain)
    end
end

function predict(bst::Booster, dmat::DMatrix,
                output_margin::Signed=0, ntree_limit::Integer=0)
    length = [1]
    ptr = XGBoosterPredict(bst, dmat, convert(Int32, output_margin),
                            convert(Uint32, ntree_limit), length)
    return pointer_to_array(ptr, XGDMatrixNumRow(dmat))
end

function set_param(bst::Booster, k::ASCIIString, v::ASCIIString)
    XGBoosterSetParam(bst, k, v)
end

function nfold_cv()

end

### save ###
function save(bst::Booster, fname::ASCIIString)
    XGBoosterSaveModel(bst, fname)
end

function save(dmat::DMatrix, fname::ASCIIString, slient::Signed)
    XGDMatrixSaveBinary(dmat, fname, convert(Int32, slient))
end

### load ###
function load(bst::Booster, fname::ASCIIString)
    XGBoosterLoadModel(bst, fname)
end

### dump model ###
function dump_model(bst::Booster, fmap::ASCIIString, out_len::Array{Integer, 1})
    XGBoosterDumpModel(bst, fmap, convert(Array{Uint64, 1}, out_len))
end

### slice ###
function slice(dmat::DMatrix, idxset::Array{Signed, 1})
    dslice = XGDMatrixSliceDMatrix(dmat, convert(Array{Int32, 1}, idxset),
                                    size(idxset)[1])
    return dslice
end

### DMatrix functions ###
function set_label(dmat::DMatrix, label::Array{Real, 1})
    @assert XGDMatrixNumRow(dmat) == size(label)[1]
    XGDMatrixSetFloatInfo(dmat, "label", convert(Array{Float32, 1}, label), size(label)[1])
end

function set_weight(dmat::DMatrix, weight::Array{Real, 1})
    @assert XGDMatrixNumRow(dmat) == size(weight)[1]
    XGDMatrixSetFloatInfo(dmat, "weight", convert(Array{Float32, 1}, weight), size(weight)[1])
end


function set_base_margin(dmat::DMatrix, margin::Array{Real, 1})
    @assert XGDMatrixNumRow(dmat) == size(margin)[1]
    XGDMatrixSetFloatInfo(dmat, "base_margin", convert(Array{Float32, 1}, margin), size(margin)[1])
end

function set_group(dmat::DMatrix, array::Array{Integer, 1}, len::Integer)
    XGDMatrixSetGroup(dmat, convert(Array{Uint32, 1}, array), convert(Uint64, len))
end

function get_label(dmat::DMatrix)
    out_len = [1]
    ptr = XGDMatrixGetFloatInfo(dmat, "label", out_len)
    return pointer_to_array(ptr, out_len[1])
end

function get_weight(dmat::DMatrix)
    out_len = [1]
    ptr = XGDMatrixGetFloatInfo(dmat, "weight", out_len)
    return pointer_to_array(ptr, out_len[1])
end

function get_margin(dmat::DMatrix)
    out_len = [1]
    ptr = XGDMatrixGetFloatInfo(dmat, "base_margin", out_len)
    return pointer_to_array(ptr, out_len[1])
end

end # module
