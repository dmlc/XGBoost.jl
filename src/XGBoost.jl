module XGBoost

include("XGBase.jl")

### train ###
function train()

end

function update()

end

function predict(bst::Booster, dmat::DMatrix,
                output_margin::Signed=0, ntree_limit::Integer=0)
    ptr = XGBoosterPredict(bst, dmat, convert(Int32, output_margin),
                            convert(Uint32, ntree_limit), length)
    # length?
    return pointer_to_array(ptr, XGDMatrixNumRow(dmat))
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

end # module
