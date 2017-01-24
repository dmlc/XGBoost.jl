include("../deps/deps.jl")

# xgboost_wrapper.h

"Calls an xgboost API function and correctly reports errors."
macro xgboost(f, params...)
    args = [param.args[1] for param in params]
    types = [param.args[2] for param in params]

    return quote
        err = ccall(($f, _xgboost), Int64, ($(types...),), $(args...))
        if err != 0
            err_msg = unsafe_string(ccall((:XGBGetLastError, _xgboost), Cstring, ()))
            error("Call to XGBoost C function ", string($f), " failed: ", err_msg)
        end
    end
end

function XGDMatrixCreateFromFile(fname::String, silent::Int32)
    handle = Ref{Ptr{Void}}()
    @xgboost(:XGDMatrixCreateFromFile,
             fname => Ptr{UInt8},
             silent => Int32,
             handle => Ref{Ptr{Void}})
    return handle[]
end

function XGDMatrixCreateFromCSC(data::SparseMatrixCSC)
    handle = Ref{Ptr{Void}}()
    @xgboost(:XGDMatrixCreateFromCSC,
             convert(Array{UInt64, 1}, data.colptr - 1) => Ptr{UInt64},
             convert(Array{UInt32, 1}, data.rowval - 1) => Ptr{UInt32},
             convert(Array{Float32, 1}, data.nzval) => Ptr{Float32},
             convert(UInt64, size(data.colptr)[1]) => UInt64,
             convert(UInt64, nnz(data)) => UInt64,
             handle => Ref{Ptr{Void}})
    return handle[]
end

function XGDMatrixCreateFromCSCT(data::SparseMatrixCSC)
    handle = Ref{Ptr{Void}}()
    @xgboost(:XGDMatrixCreateFromCSR,
             convert(Array{UInt64, 1}, data.colptr - 1) => Ptr{UInt64},
             convert(Array{UInt32, 1}, data.rowval - 1) => Ptr{UInt32},
             convert(Array{Float32, 1}, data.nzval) => Ptr{Float32},
             convert(UInt64, size(data.colptr)[1]) => UInt64,
             convert(UInt64, nnz(data)) => UInt64,
             handle => Ref{Ptr{Void}})
    return handle[]
end

function XGDMatrixCreateFromMat(data::Array{Float32,2}, missing::Float32)
    XGDMatrixCreateFromMatT(transpose(data), missing)
end

function XGDMatrixCreateFromMatT(data::Array{Float32,2}, missing::Float32)
    ncol, nrow = size(data)
    handle = Ref{Ptr{Void}}()
    @xgboost(:XGDMatrixCreateFromMat,
             data => Ptr{Float32},
             nrow => UInt64,
             ncol => UInt64,
             missing => Float32,
             handle => Ref{Ptr{Void}})
    return handle[]
end

function XGDMatrixSliceDMatrix(handle::Ptr{Void}, idxset::Array{Int32,1}, len::UInt64)
    ret = Ref{Ptr{Void}}()
    @xgboost(:XGDMatrixSliceDMatrix,
             handle => Ptr{Void},
             idxset => Ptr{Int32},
             len => UInt64,
             ret => Ref{Ptr{Void}})
    return ret[]
end

function XGDMatrixFree(handle::Ptr{Void})
    @xgboost(:XGDMatrixFree,
             handle => Ptr{Void})
end

function XGDMatrixSaveBinary(handle::Ptr{Void}, fname::String, silent::Int32)
    @xgboost(:XGDMatrixSaveBinary,
             handle => Ptr{Void},
             fname => Ptr{UInt8},
             silent => Int32)
end

function XGDMatrixSetFloatInfo(handle::Ptr{Void}, field::String, array::Array{Float32,1},
                               len::UInt64)
    @xgboost(:XGDMatrixSetFloatInfo,
             handle => Ptr{Void},
             field => Ptr{UInt8},
             array => Ptr{Float32},
             len => UInt64)
end

function XGDMatrixSetUIntInfo(handle::Ptr{Void}, field::String, array::Array{UInt32,1},
                              len::UInt64)
    @xgboost(:XGDMatrixSetUIntInfo,
             handle => Ptr{Void},
             field => Ptr{UInt8},
             array => Ptr{UInt32},
             len => UInt64)
end

function XGDMatrixSetGroup(handle::Ptr{Void}, array::Array{UInt32,1}, len::UInt64)
    @xgboost(:XGDMatrixSetGroup,
             handle => Ptr{Void},
             array => Ptr{UInt32},
             len => UInt64)
end

function XGDMatrixGetFloatInfo(handle::Ptr{Void}, field::String, outlen::Array{UInt64,1})
    ret = Ref{Ptr{Float32}}()
    @xgboost(:XGDMatrixGetFloatInfo,
             handle => Ptr{Void},
             field => Ptr{UInt8},
             outlen => Ptr{UInt64},
             ret =>  Ref{Ptr{Float32}})
    return ret[]
end

function XGDMatrixGetUIntInfo(handle::Ptr{Void}, field::String, outlen::Array{UInt64,1})
    ret = Ref{Ptr{UInt32}}()
    @xgboost(:XGDMatrixGetUIntInfo,
             handle => Ptr{Void},
             field => Ptr{UInt8},
             outlen => Ptr{UInt64},
             ret => Ref{Ptr{UInt32}})
    return ret[]
end

function XGDMatrixNumRow(handle::Ptr{Void})
    ret = Ref{UInt64}()
    @xgboost(:XGDMatrixNumRow,
             handle => Ptr{Void},
             ret => Ref{UInt64})
    return ret[]
end

function JLGetFloatInfo(handle::Ptr{Void}, field::String)
    len = UInt64[1]
    ptr = XGDMatrixGetFloatInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function JLGetUintInfo(handle::Ptr{Void}, field::String)
    len = UInt64[1]
    ptr = XGDMatrixGetUIntInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function XGBoosterCreate(cachelist::Array{Ptr{Void},1}, len::Int64)
    handle = Ref{Ptr{Void}}()
    @xgboost(:XGBoosterCreate,
             cachelist => Ptr{Ptr{Void}},
             len => UInt64,
             handle => Ref{Ptr{Void}})
    return handle[]
end

function XGBoosterFree(handle::Ptr{Void})
    @xgboost(:XGBoosterFree,
             handle => Ptr{Void})
end

function XGBoosterSetParam(handle::Ptr{Void}, key::String, value::String)
    @xgboost(:XGBoosterSetParam,
             handle => Ptr{Void},
             key => Ptr{UInt8},
             value => Ptr{UInt8})
end

function XGBoosterUpdateOneIter(handle::Ptr{Void}, iter::Int32, dtrain::Ptr{Void})
    @xgboost(:XGBoosterUpdateOneIter,
             handle => Ptr{Void},
             iter => Int32,
             dtrain => Ptr{Void})
end

function XGBoosterBoostOneIter(handle::Ptr{Void}, dtrain::Ptr{Void}, grad::Array{Float32,1},
                               hess::Array{Float32,1}, len::UInt64)
    @xgboost(:XGBoosterBoostOneIter,
             handle => Ptr{Void},
             dtrain => Ptr{Void},
             grad => Ptr{Float32},
             hess => Ptr{Float32},
             len => UInt64)
end

function XGBoosterEvalOneIter(handle::Ptr{Void}, iter::Int32, dmats::Array{Ptr{Void},1},
                              evnames::Array{String,1}, len::UInt64)
    msg = Ref{Ptr{UInt8}}()
    @xgboost(:XGBoosterEvalOneIter,
             handle => Ptr{Void},
             iter => Int32,
             dmats => Ptr{Ptr{Void}},
             evnames => Ptr{Ptr{UInt8}},
             len => UInt64,
             msg => Ref{Ptr{UInt8}})
    return unsafe_string(msg[])
end

function XGBoosterPredict(handle::Ptr{Void}, dmat::Ptr{Void}, output_margin::Int32,
                          ntree_limit::UInt32, len::Array{UInt64,1})
    ret = Ref{Ptr{Float32}}()
    @xgboost(:XGBoosterPredict,
             handle => Ptr{Void},
             dmat => Ptr{Void},
             output_margin => Int32,
             ntree_limit => UInt32,
             len => Ptr{UInt64},
             ret => Ref{Ptr{Float32}})
    return ret[]
end

function XGBoosterLoadModel(handle::Ptr{Void}, fname::String)
    @xgboost(:XGBoosterLoadModel,
             handle => Ptr{Void},
             fname => Ptr{UInt8})
end

function XGBoosterSaveModel(handle::Ptr{Void}, fname::String)
    @xgboost(:XGBoosterSaveModel,
             handle => Ptr{Void},
             fname => Ptr{UInt8})
end

function XGBoosterDumpModel(handle::Ptr{Void}, fmap::String, with_stats::Int64)
    data = Ref{Ptr{Ptr{UInt8}}}()
    out_len = Ref{UInt64}(0)
    @xgboost(:XGBoosterDumpModel,
             handle => Ptr{Void},
             fmap => Ptr{UInt8},
             with_stats => Int64,
             out_len => Ref{UInt64},
             data => Ref{Ptr{Ptr{UInt8}}})
    return unsafe_wrap(Array, data[], out_len[])
end
