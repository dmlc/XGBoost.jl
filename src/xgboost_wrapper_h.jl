include("../deps/deps.jl")

# xgboost_wrapper.h

"Calls an xgboost API function and correctly reports errors."
macro xgboost_ccall(f, argTypes, args...)
    argTypes = eval(argTypes)
    return quote
        err = ccall(($f, _xgboost), Int64, ($(argTypes...),), $(args...))
        if err != 0
            errMsg = bytestring(ccall((:XGBGetLastError, _xgboost), Ptr{UInt8}, ()))
            error("Call to XGBoost C function "*string($f)*" failed: $errMsg")
        end
    end
end

function XGDMatrixCreateFromFile(fname::ASCIIString, slient::Int32)
    handle = Ref{Ptr{Void}}()
    @xgboost_ccall(
        :XGDMatrixCreateFromFile,
        (Ptr{UInt8}, Int32, Ref{Ptr{Void}}),
        fname, slient, handle
    )
    return handle[]
end

function XGDMatrixCreateFromCSC(data::SparseMatrixCSC{Float32, Int64})
    handle = Ref{Ptr{Void}}()
    @xgboost_ccall(
        :XGDMatrixCreateFromCSC,
        (Ptr{UInt64}, Ptr{UInt32}, Ptr{Float32}, UInt64, UInt64, Ref{Ptr{Void}}),
        convert(Array{UInt64, 1}, data.colptr - 1),
        convert(Array{UInt32, 1}, data.rowval - 1), data.nzval,
        convert(UInt64, size(data.colptr)[1]),
        convert(UInt64, nnz(data)),
        handle
    )
    return handle[]
end

function XGDMatrixCreateFromMat(data::Array{Float32, 2}, missing::Float32)
    data = transpose(data)
    nrow = size(data)[2]
    ncol = size(data)[1]
    handle = Ref{Ptr{Void}}()
    @xgboost_ccall(
        :XGDMatrixCreateFromMat,
        (Ptr{Float32}, UInt64, UInt64, Float32, Ref{Ptr{Void}}),
        data, nrow, ncol, missing, handle
    )
    return handle[]
end

function XGDMatrixSliceDMatrix(handle::Ptr{Void}, idxset::Array{Int32, 1}, len::UInt64)
    ret = Ref{Ptr{Void}}()
    @xgboost_ccall(
        :XGDMatrixSliceDMatrix,
        (Ptr{Void}, Ptr{Int32}, UInt64, Ref{Ptr{Void}}),
        handle, idxset, len, ret
    )
    return ret[]
end

function XGDMatrixFree(handle::Ptr{Void})
    @xgboost_ccall(
        :XGDMatrixFree,
        (Ptr{Void},),
        handle
    )
end

function XGDMatrixSaveBinary(handle::Ptr{Void}, fname::ASCIIString, slient::Int32)
    @xgboost_ccall(
        :XGDMatrixSaveBinary,
        (Ptr{Void}, Ptr{UInt8}, Int32),
        handle, fname, slient
    )
end

function XGDMatrixSetFloatInfo(handle::Ptr{Void}, field::ASCIIString,
                               array::Array{Float32, 1}, len::UInt64)
    @xgboost_ccall(
        :XGDMatrixSetFloatInfo,
        (Ptr{Void}, Ptr{UInt8}, Ptr{Float32}, UInt64),
        handle, field, array, len
    )
end

function XGDMatrixSetUIntInfo(handle::Ptr{Void}, field::ASCIIString,
                              array::Array{UInt32, 1}, len::UInt64)
    @xgboost_ccall(
        :XGDMatrixSetUIntInfo,
        (Ptr{Void}, Ptr{UInt8}, Ptr{UInt32}, UInt64),
        handle, field, array, len
    )
end

function XGDMatrixSetGroup(handle::Ptr{Void}, array::Array{UInt32, 1}, len::UInt64)
    @xgboost_ccall(
        :XGDMatrixSetGroup,
        (Ptr{Void}, Ptr{UInt32}, UInt64),
         handle, array, len
    )
end

function XGDMatrixGetFloatInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{UInt64, 1})
    ret = Ref{Ptr{Float32}}()
    @xgboost_ccall(
        :XGDMatrixGetFloatInfo,
        (Ptr{Void}, Ptr{UInt8}, Ptr{UInt64}, Ref{Ptr{Float32}}),
         handle, field, outlen, ret
    )
    return ret[]
end

function XGDMatrixGetUIntInfo(handle::Ptr{Void}, field::ASCIIString, outlen::Array{UInt64, 1})
    ret = Ref{Ptr{UInt32}}()
    @xgboost_ccall(
        :XGDMatrixGetUIntInfo,
        (Ptr{Void}, Ptr{UInt8}, Ptr{UInt64}, Ref{Ptr{UInt32}}),
         handle, field, outlen, ret
    )
    return ret[]
end

function XGDMatrixNumRow(handle::Ptr{Void})
    ret = Ref{UInt64}()
    @xgboost_ccall(
        :XGDMatrixNumRow,
        (Ptr{Void}, Ref{UInt64}),
         handle, ret
    )
    return ret[]
end

function JLGetFloatInfo(handle::Ptr{Void}, field::ASCIIString)
    len = UInt64[1]
    ptr = XGDMatrixGetFloatInfo(handle, field, len)
    return pointer_to_array(ptr, len[1])
end

function JLGetUintInfo(handle::Ptr{Void}, field::ASCIIString)
    len = UInt64[1]
    ptr = XGDMatrixGetUIntInfo(handle, field, len)
    return pointer_to_array(ptr, len[1])
end

function XGBoosterCreate(cachelist::Array{Ptr{Void}, 1}, len::Int64)
    handle = Ref{Ptr{Void}}()
    @xgboost_ccall(
        :XGBoosterCreate,
        (Ptr{Ptr{Void}}, UInt64, Ref{Ptr{Void}}),
        cachelist, len, handle
    )
    return handle[]
end

function XGBoosterFree(handle::Ptr{Void})
    @xgboost_ccall(
        :XGBoosterFree,
        (Ptr{Void}, ),
        handle
    )
end

function XGBoosterSetParam(handle::Ptr{Void}, key::ASCIIString, value::ASCIIString)
    @xgboost_ccall(
        :XGBoosterSetParam,
        (Ptr{Void}, Ptr{UInt8}, Ptr{UInt8}),
        handle, key, value
    )
end

function XGBoosterUpdateOneIter(handle::Ptr{Void}, iter::Int32, dtrain::Ptr{Void})
    @xgboost_ccall(
        :XGBoosterUpdateOneIter,
        (Ptr{Void}, Int32, Ptr{Void}),
        handle, iter, dtrain
    )
end

function XGBoosterBoostOneIter(handle::Ptr{Void}, dtrain::Ptr{Void},
                               grad::Array{Float32, 1},
                               hess::Array{Float32, 1},
                               len::UInt64)
    @xgboost_ccall(
        :XGBoosterBoostOneIter,
        (Ptr{Void}, Ptr{Void}, Ptr{Float32}, Ptr{Float32}, UInt64),
        handle, dtrain, grad, hess, len
    )
end

function XGBoosterEvalOneIter(handle::Ptr{Void}, iter::Int32,
                              dmats::Array{Ptr{Void}, 1},
                              evnames::Array{ASCIIString, 1}, len::UInt64)
    msg = Ref{Ptr{UInt8}}()
    @xgboost_ccall(
        :XGBoosterEvalOneIter,
        (Ptr{Void}, Int32, Ptr{Ptr{Void}}, Ptr{Ptr{UInt8}}, UInt64, Ref{Ptr{UInt8}}),
        handle, iter, dmats, evnames, len, msg
    )
    return bytestring(msg[])
end


function XGBoosterPredict(handle::Ptr{Void}, dmat::Ptr{Void}, output_margin::Int32,
                          ntree_limit::UInt32, len::Array{UInt64, 1})
    ret = Ref{Ptr{Float32}}()
    @xgboost_ccall(
        :XGBoosterPredict,
        (Ptr{Void}, Ptr{Void}, Int32, UInt32, Ptr{UInt64}, Ref{Ptr{Float32}}),
        handle, dmat, output_margin, ntree_limit, len, ret
    )
    return ret[]
end


function XGBoosterLoadModel(handle::Ptr{Void}, fname::ASCIIString)
    @xgboost_ccall(
        :XGBoosterLoadModel,
        (Ptr{Void}, Ptr{UInt8}),
        handle, fname
    )
end

function XGBoosterSaveModel(handle::Ptr{Void}, fname::ASCIIString)
    @xgboost_ccall(
        :XGBoosterSaveModel,
        (Ptr{Void}, Ptr{UInt8}),
        handle, fname
    )
end


function XGBoosterDumpModel(handle::Ptr{Void}, fmap::ASCIIString, with_stats::Int64)
    data = Ref{Ptr{Ptr{UInt8}}}()
    out_len = Ref{UInt64}(0)
    @xgboost_ccall(
        :XGBoosterDumpModel,
        (Ptr{Void}, Ptr{UInt8}, Int64, Ref{UInt64}, Ref{Ptr{Ptr{UInt8}}}),
        handle, fmap, with_stats, out_len, data
    )
    return pointer_to_array(data[], out_len[])
end
