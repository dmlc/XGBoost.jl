include("../deps/deps.jl")

# xgboost_wrapper.h
#
function XGDMatrixCreateFromFile(fname::ASCIIString, slient::Int32)
    handle = ccall((:XGDMatrixCreateFromFile,
                    _xgboost
                    ), 
                   Ptr{Void}, (Ptr{Uint8}, Int32),
                   fname, slient)
    return handle
end

function XGDMatrixCreateFromCSC(data::SparseMatrixCSC{Float32, Int64})
    handle = ccall((:XGDMatrixCreateFromCSC, _xgboost),
                   Ptr{Void},
                   (Ptr{Uint64}, Ptr{Uint32}, Ptr{Float32}, Uint64, Uint64),
                   convert(Array{Uint64, 1}, data.colptr - 1),
                   convert(Array{Uint32, 1}, data.rowval - 1), data.nzval,
                   convert(Uint64, size(data.colptr)[1]),
                   convert(Uint64, nnz(data)))
    return handle
end

function XGDMatrixCreateFromMat(data::Array{Float32, 2}, missing::Float32)
    nrow = size(data)[1]
    ncol = size(data)[2]
    handle = ccall((:XGDMatrixCreateFromMat, _xgboost),
                   Ptr{Void},
                   (Ptr{Float32}, Uint64, Uint64, Float32),
                   data, nrow, ncol, missing)
    return handle
end

function XGDMatrixSliceDMatrix(handle::Ptr{Void}, idxset::Array{Int32, 1}, len::Uint64)
    ret = ccall((:XGDMatrixSliceDMatrix, _xgboost),
                Ptr{Void},
                (Ptr{Void}, Ptr{Int32}, Uint64),
                handle, idxset, len)
    return ret
end

function XGDMatrixFree(handle::Ptr{Void})
    ccall((:XGDMatrixFree, _xgboost),
          Void,
          (Ptr{Void}, ),
          handle)
end

function XGDMatrixSaveBinary(handle::Ptr{Void}, fname::ASCIIString, slient::Int32)
    ccall((:XGDMatrixSaveBinary, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Int32),
          handle, fname, slient)
end

function XGDMatrixSetFloatInfo(handle::Ptr{Void}, field::ASCIIString,
                               array::Array{Float32, 1}, len::Uint64)
    ccall((:XGDMatrixSetFloatInfo, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Float32}, Uint64),
          handle, field, array, len)
end

function XGDMatrixSetUIntInfo(handle::Ptr{Void}, field::ASCIIString,
                              array::Array{Uint32, 1}, len::Uint64)
    ccall((:XGDMatrixSetUIntInfo, _xgboost),
    Void,
    (Ptr{Void}, Ptr{Uint8}, Ptr{Uint32}, Uint64),
    handle, field, array, len)
end

function XGDMatrixSetGroup(handle::Ptr{Void}, array::Array{Uint32, 1}, len::Uint64)
    ccall((:XGDMatrixSetGroup, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Uint32}, Uint64),
          handle, array, len)
end

function XGDMatrixGetFloatInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetFloatInfo, _xgboost),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, field, outlen)
end

function XGDMatrixGetUIntInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetUIntInfo, _xgboost),
                 Ptr{Uint32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, field, outlen)
end

function XGDMatrixNumRow(handle::Ptr{Void})
    return ccall((:XGDMatrixNumRow, _xgboost),
                 Uint64,
                 (Ptr{Void},),
                 handle)
end

function JLGetFloatInfo(handle::Ptr{Void}, field::ASCIIString)
    len = Uint64[1]
    ptr = XGDMatrixGetFloatInfo(handle, field, len)
    return pointer_to_array(ptr, len[1])
end

function JLGetUintInfo(handle::Ptr{Void}, field::ASCIIString)
    len = Uint64[1]
    ptr = XGDMatrixGetUIntInfo(handle, field, len)
    return pointer_to_array(ptr, len[1])
end

function XGBoosterCreate(cachelist::Array{Ptr{Void}, 1}, len::Int64)
    handle = ccall((:XGBoosterCreate, _xgboost),
                   Ptr{Void},
                   (Ptr{Ptr{Void}}, Uint64),
                   cachelist, len)
    return handle
end

function XGBoosterFree(handle::Ptr{Void})
    ccall((:XGBoosterFree, _xgboost),
          Void,
          (Ptr{Void}, ),
          handle)
end

function XGBoosterSetParam(handle::Ptr{Void}, key::ASCIIString, value::ASCIIString)
    ccall((:XGBoosterSetParam, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
          handle, key, value)
end

function XGBoosterUpdateOneIter(handle::Ptr{Void}, iter::Int32, dtrain::Ptr{Void})
    ccall((:XGBoosterUpdateOneIter, _xgboost),
          Void,
          (Ptr{Void}, Int32, Ptr{Void}),
          handle, iter, dtrain)
end

function XGBoosterBoostOneIter(handle::Ptr{Void}, dtrain::Ptr{Void},
                               grad::Array{Float32, 1},
                               hess::Array{Float32, 1},
                               len::Uint64)
    ccall((:XGBoosterBoostOneIter, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Void}, Ptr{Float32}, Ptr{Float32}, Uint64),
          handle, dtrain, grad, hess, len)
end

function XGBoosterEvalOneIter(handle::Ptr{Void}, iter::Int32,
                              dmats::Array{Ptr{Void}, 1},
                              evnames::Array{ASCIIString, 1}, len::Uint64)
    msg = ccall((:XGBoosterEvalOneIter, _xgboost),
                Ptr{Uint8},
                (Ptr{Void}, Int32, Ptr{Ptr{Void}}, Ptr{Ptr{Uint8}}, Uint64),
                handle, iter, dmats, evnames, len)
    return bytestring(msg)
end


function XGBoosterPredict(handle::Ptr{Void}, dmat::Ptr{Void}, output_margin::Int32,
                          ntree_limit::Uint32, len::Array{Uint64, 1})
    ptr = ccall((:XGBoosterPredict, _xgboost),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Void}, Int32, Uint32, Ptr{Uint64}),
                 handle, dmat, output_margin, ntree_limit, len)
    return ptr
end


function XGBoosterLoadModel(handle::Ptr{Void}, fname::ASCIIString)
    ccall((:XGBoosterLoadModel, _xgboost),
          Void,
          (Ptr{Void}, Ptr{Uint8}),
          handle, fname)
end

function XGBoosterSaveModel(handle::Ptr{Void}, fname::ASCIIString)
    ccall((:XGBoosterSaveModel, _xgboost),
           Void,
          (Ptr{Void}, Ptr{Uint8}),
          handle, fname)
end


function XGBoosterDumpModel(handle::Ptr{Void}, fmap::ASCIIString, out_len::Array{Uint64, 1})
    data = ccall((:XGBoosterDumpModel, _xgboost),
                  Ptr{Ptr{Uint8}},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, fmap, out_len)
    
    return data
end

