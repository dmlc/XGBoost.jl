### DMatrix ###
type DMatrix
    handle::Ptr{Void}
    function DMatrix(fname::ASCIIString, slient::Int32)
        handle = ccall((:XGDMatrixCreateFromFile,
                        "../xgboost/wrapper/libxgboostwrapper.so"),
                       Ptr{Void},
                       (Ptr{Uint8}, Int32),
                       fname, slient)
        new(handle)
    end
    function DMatrix(data::SparseMatrixCSC{Float32, Int64})
        handle = ccall((:XGDMatrixCreateFromCSC,
                        "../xgboost/wrapper/libxgboostwrapper.so"),
                       Ptr{Void},
                       (Ptr{Uint64}, Ptr{Uint32}, Ptr{Float32}, Uint64, Uint64),
                       csr.colptr - 1, csr.rowval - 1, csr.nzval, size(csr.colptr)[1], nnz(csr))
        new(handle)
    end
    function DMatrix(data::Array{Float32, 2}, missing::Float32)
        nrow = size(data)[1]
        ncol = size(data)[2]
        handle = ccall((:XGDMatrixCreateFromMat,
                        "../xgboost/wrapper/libxgboostwrapper.so"),
                       Ptr{Void},
                       (Ptr{Float32}, Uint64, Uint64, Float32),
                       data, nrow, ncol, missing)
        new(handle)
    end
end

function XGDMatrixSliceDMatrix(dmat::DMatrix, idxset::Array{Int32, 1}, len::Uint64)
    handle = ccall((:XGDMatrixSliceDMatrix,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Void}, Ptr{Int32}, Uint64),
                   dmat.handle, idxset, len)
    return DMatrix(handle)
end


function XGDMatrixFree(dmat::DMatrix)
    ccall((:XGDMatrixFree,
        "../xgboost/wrapper/libxgboostwrapper.so"),
        Void,
        (Ptr{Void}, ),
        dmat.handle)
end

function XGDMatrixSaveBinary(dmat::DMatrix, fname::ASCIIString, slient::Int32)
    ccall((:XGDMatrixSaveBinary,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Int32),
          dmat.handle, fname, slient)
end

function XGDMatrixSetFloatInfo(dmat::DMatrix, field::ASCIIString,
                               array::Array{Float32, 1}, len::Uint64)
    ccall((:XGDMatrixSetFloatInfo,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Float32}, Uint64),
          dmat.handle, field, array, len)
end

function XGDMatrixSetUIntInfo(dmat::DMatrix, field::ASCIIString,
                              array::Array{Uint32, 1}, len::Uint64))
     ccall((:XGDMatrixSetUIntInfo,
             "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Uint32}, Uint64),
          dmat.handle, field, array, len)
end

function XGDMatrixSetGroup(dmat::DMatrix, array::Array{Uint32, 1}, len::Uint64)
    ccall((:XGDMatrixSetGroup,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint32}, Uint64),
          dmat.handle, array, len)
end

function XGDMatrixGetFloatInfo(dmat::DMatrix, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetFloatInfo,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 dmat.handle, field, outlen)
end

function XGDMatrixGetUIntInfo(dmat::DMatrix, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetUIntInfo,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Uint32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 dmat.handle, field, outlen)
end

function XGDMatrixNumRow(dmat::DMatrix)
    return ccall((:XGDMatrixNumRow,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Uint64,
                 (Ptr{Ptr{Void}},)
                 dmat.handle)
end


### Booster ###
type Booster
    handle::Ptr{Void}
    function Booster(dmats::Array{DMatrix, 1}, len::Int64)
        handle = ccall((:XGBoosterCreate,
                        "../xgboost/wrapper/libxgboostwrapper.so"),
                       Ptr{Void},
                       (Ptr{Ptr{Void}}, Culong),
                       [itm.handle for itm in dmats], len)
        new(handle)
    end
end

function XGBoosterFree(bst::Booster)
    ccall((:XGBoosterFree,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, ),
          bst.handle)
end

function XGBoosterSetParam(bst::Booster, key::ASCIIString, value::ASCIIString)
    ccall((:XGBoosterSetParam,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
          bst.handle, key, value)
end

function XGBoosterUpdateOneIter(bst::Booster, iter::Int32, dtrain::DMatrix)
    ccall((:XGBoosterUpdateOneIter,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Int32, Ptr{Void}),
          bst.handle, iter, dtrain.handle)
end


function XGBoosterBoostOneIter(bst::Booster, iter::Int32,
                               grad::Array{Float32, 1},
                               hess::Array{Float32, 1},
                               len::Uint64)
    ccall((:XGBoosterBoostOneIter,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Int32, Ptr{Float32}, Ptr{Float32}, Uint64),
          bst.handle, iter, grad, hess, len)
end

function XGBoosterEvalOneIter(bst::Booster, iter::Int32, dmats::Array{DMatrix, 1},
                              evnames::Array{ASCIIString, 1}, len::Uint64)
    msg = ccall((:XGBoosterEvalOneIter,
                 "../xgboost/wrapper/libxgboostwrapper.so"),
                Ptr{Uint8},
                (Ptr{Void}, Int32, Ptr{Ptr{Void}}, Ptr{Uint8}, Uint64),
                bst.handle, iter, [itm.handle for itm in dmats], evnames, len)
    return msg
end

function XGBoosterPredict(bst::Booster, dmat::DMatrix, output_margin::Int32,
                          ntree_limit::Uint32, len::Array{Uint64, 1})
    pred = ccall((:XGBoosterPredict,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Void}, Int32, Uint32, Ptr{Uint64}),
                 bst.handle, dmat.handle, output_margin, ntree_limit, len)
    return pred
end

function XGBoosterLoadModel(bst::Booster, fname::ASCIIString)
    ccall((:XGBoosterLoadModel,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}),
          bst.handle, fname)
end

function XGBoosterSaveModel(bst::Booster, fname::ASCIIString)
    ccall((:XGBoosterSaveModel,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          (Ptr{Void}, Ptr{Uint8}),
          bst.handle, fname)
end

function XGBoosterDumpModel(bst::Booster, fmap::ASCIIString, out_len::Array{Uint64, 1})
    data = ccall((:XGBoosterDumpModel,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 bst.handle, fmap, out_len)
    return data
end
