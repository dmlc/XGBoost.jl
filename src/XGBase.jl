
function XGDMatrixCreateFromFile(fname::ASCIIString, slient::Int32)
    handle = ccall((:XGDMatrixCreateFromFile,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Uint8}, Int32),
                   fname, slient)
    return handle
end

function XGDMatrixCreateFromCSC(data::SparseMatrixCSC{Float32, Int64})
    handle = ccall((:XGDMatrixCreateFromCSC,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Uint64}, Ptr{Uint32}, Ptr{Float32}, Uint64, Uint64),
                   data.colptr - 1, data.rowval - 1, data.nzval, size(data.colptr)[1], nnz(data))
    return handle
end

function XGDMatrixCreateFromMat(data::Array{Float32, 2}, missing::Float32)
    nrow = size(data)[1]
    ncol = size(data)[2]
    handle = ccall((:XGDMatrixCreateFromMat,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Float32}, Uint64, Uint64, Float32),
                   data, nrow, ncol, missing)
    return handle
end

function XGDMatrixSliceDMatrix(handle::Ptr{Void}, idxset::Array{Int32, 1}, len::Uint64)
    handle = ccall((:XGDMatrixSliceDMatrix,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Void}, Ptr{Int32}, Uint64),
                   dmat.handle, idxset, len)
    return handle
end

function XGDMatrixFree(handle::Ptr{Void})
    ccall((:XGDMatrixFree,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, ),
          handle)
end

function XGDMatrixSaveBinary(handle::Ptr{Void}, fname::ASCIIString, slient::Int32)
    ccall((:XGDMatrixSaveBinary,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Int32),
          handle, fname, slient)
end

function XGDMatrixSetFloatInfo(handle::Ptr{Void}, field::ASCIIString,
                               array::Array{Float32, 1}, len::Uint64)
    ccall((:XGDMatrixSetFloatInfo,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Float32}, Uint64),
          handle, field, array, len)
end

function XGDMatrixSetUIntInfo(handle::Ptr{Void}, field::ASCIIString,
                              array::Array{Uint32, 1}, len::Uint64)
    ccall((:XGDMatrixSetUIntInfo,
           "../xgboost/wrapper/libxgboostwrapper.so"),
    Void,
    (Ptr{Void}, Ptr{Uint8}, Ptr{Uint32}, Uint64),
    handle, field, array, len)
end

function XGDMatrixSetGroup(handle::Ptr{Void}, array::Array{Uint32, 1}, len::Uint64)
    ccall((:XGDMatrixSetGroup,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint32}, Uint64),
          handle, array, len)
end

function XGDMatrixGetFloatInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetFloatInfo,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, field, outlen)
end

function XGDMatrixGetUIntInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{Uint64, 1})
    return ccall((:XGDMatrixGetUIntInfo,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Uint32},
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, field, outlen)
end



function XGDMatrixNumRow(handle::Ptr{Void})
    return ccall((:XGDMatrixNumRow,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
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

type DMatrix
    handle::Ptr{Void}
    function _SetMeta(handle::Ptr{Void}, weight, group, label, margin)
        if typeof(weight) == Array{Float64, 1} ||
            typeof(weight) == Array{Float32, 1}
            @assert XGDMatrixNumRow(handle) == size(weight)[1]
            XGDMatrixGetFloatInfo(ptr, "weight",
                                  convert(Array{Float32, 1}, weight),
                                  size(weight)[1])
        end
        if typeof(group) == Array{Int64, 1} || typeof(weight) == Array{Int32, 1}
            @assert XGDMatrixNumRow(handle) == size(group)[1]
            XGDMatrixSetGroup(handle, convert(Array{Uint32, 1}, group, size(group)[1]))
        end
        if typeof(label) == Array{Float64, 1} || typeof(label) == Array{Float32, 1}
            @assert XGDMatrixNumRow(handle) == size(label)[1]
            XGDMatrixGetFloatInfo(ptr, "label",
                                  convert(Array{Float32, 1}, label),
                                  size(label)[1])
        end
        if typeof(margin) == Array{Float64, 1} || typeof(margin) == Array{Float32, 1}
            @assert XGDMatrixNumRow(handle) == size(margin)[1]
            XGDMatrixGetFloatInfo(ptr, "base_margin",
                                  convert(Array{Float32, 1}, margin),
                                  size(margin)[1])
        end
    end
    function DMatrix(handle::Ptr{Void})
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(fname::ASCIIString; slient::Integer=0, weight=None, group=None, margin=None)
        handle = XGDMatrixCreateFromFile(fname, convert(Int32, slient))
        _SetMeta(handle, weight, group, None, margin)
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(data::SparseMatrixCSC{Float32, Int64}; label=None, weight=None, group=None, margin=None)
        handle = XGDMatrixCreateFromCSC(data)
        _SetMeta(handle, weight, group, label, margin)
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
    function DMatrix(data::Array{Float32, 2}; missing::Float32=0,
                     label=None, weight=None, group=None, margin=None)
        handle = XGDMatrixCreateFromMat(data)
        _SetMeta(handle, weight, group, label, margin)
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
end


function XGBoosterCreate(dmats::Array{DMatrix, 1}, len::Int64)
    handle = ccall((:XGBoosterCreate,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                   Ptr{Void},
                   (Ptr{Ptr{Void}}, Uint64),
                   [itm.handle for itm in dmats], len)
    return handle
end

function XGBoosterFree(handle::Ptr{Void})
    ccall((:XGBoosterFree,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, ),
          handle)
end

function XGBoosterSetParam(handle::Ptr{Void}, key::ASCIIString, value::ASCIIString)
    ccall((:XGBoosterSetParam,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
          handle, key, value)
end

function XGBoosterUpdateOneIter(handle::Ptr{Void}, iter::Int32, dtrain::DMatrix)
    ccall((:XGBoosterUpdateOneIter,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Int32, Ptr{Void}),
          handle, iter, dtrain.handle)
end

function XGBoosterBoostOneIter(handle::Ptr{Void}, iter::Int32,
                               grad::Array{Float32, 1},
                               hess::Array{Float32, 1},
                               len::Uint64)
    ccall((:XGBoosterBoostOneIter,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Int32, Ptr{Float32}, Ptr{Float32}, Uint64),
          handle, iter, grad, hess, len)
end

function XGBoosterEvalOneIter(handle::Ptr{Void}, iter::Int32,
                              dmats::Array{DMatrix, 1},
                              evnames::Array{ASCIIString, 1}, len::Uint64)
    msg = ccall((:XGBoosterEvalOneIter,
                 "../xgboost/wrapper/libxgboostwrapper.so"),
                Ptr{Uint8},
                (Ptr{Void}, Int32, Ptr{Ptr{Void}}, Ptr{Ptr{Uint8}}, Uint64),
                handle, iter, [itm.handle for itm in dmats], evnames, len)
    return bytestring(msg)
end


function XGBoosterPredict(handle::Ptr{Void}, dmat::DMatrix, output_margin::Int32,
                          ntree_limit::Uint32, len::Array{Uint64, 1})
    ptr = ccall((:XGBoosterPredict,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                 Ptr{Float32},
                 (Ptr{Void}, Ptr{Void}, Int32, Uint32, Ptr{Uint64}),
                 handle, dmat.handle, output_margin, ntree_limit, len)
    return ptr
end


function XGBoosterLoadModel(handle::Ptr{Void}, fname::ASCIIString)
    ccall((:XGBoosterLoadModel,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}),
          handle, fname)
end

function XGBoosterSaveModel(handle::Ptr{Void}, fname::ASCIIString)
    ccall((:XGBoosterSaveModel,
           "../xgboost/wrapper/libxgboostwrapper.so"),
           Void,
          (Ptr{Void}, Ptr{Uint8}),
          handle, fname)
end


### Check later
function XGBoosterDumpModel(handle::Ptr{Void}, fmap::ASCIIString, out_len::Array{Uint64, 1})
    data = ccall((:XGBoosterDumpModel,
                  "../xgboost/wrapper/libxgboostwrapper.so"),
                  Void,
                 (Ptr{Void}, Ptr{Uint8}, Ptr{Uint64}),
                 handle, fmap, out_len)
    return data
end


type Booster
    handle::Ptr{Void}
    function Booster(dmats::Array{DMatrix, 1}, len::Int64)
        handle = XGBoosterCreate(dmats::Array{DMatrix, 1}, len::Int64)
        new(handle)
    end
    function Booster(fname::ASCIIString)
        handle = XGBoosterCreate(DMatrix[], 0)
        XGBoosterLoadModel(handle, fname)
        sp = new(handle)
        finalizer(sp, JLFree)
        sp
    end
end

function JLFree(bst::Booster)
    XGBoosterFree(bst.handle)
end

function JLFree(dmat::DMatrix)
    XGDMatrixFree(dmat.handle)
end
