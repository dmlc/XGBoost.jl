include("../deps/deps.jl")

if build_version == "master"
    typealias Bst_ulong Culonglong
else
    typealias Bst_ulong Culong
end

typealias DMatrixHandle Ptr{Void}
typealias BoosterHandle Ptr{Void}

"Calls an xgboost API function and correctly reports errors."
macro xgboost(f, params...)
    return quote
        err = ccall(($f, _jl_libxgboost), Int64,
                    ($((esc(i.args[2]) for i in params)...),),
                    $((esc(i.args[1]) for i in params)...))
        if err != 0
            err_msg = unsafe_string(ccall((:XGBGetLastError, _jl_libxgboost), Cstring, ()))
            error("Call to XGBoost C function ", string($(esc(f))), " failed: ", err_msg)
        end
    end
end

function XGDMatrixCreateFromFile(fname::String, silent::Int32)
    out = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromFile,
             fname => Cstring,
             silent => Cint,
             out => Ref{DMatrixHandle})
    return out[]
end

function XGDMatrixCreateFromCSC(data::SparseMatrixCSC)
    out = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromCSC,
             convert(Vector{UInt64}, data.colptr - 1) => Ptr{Bst_ulong},
             convert(Vector{UInt32}, data.rowval - 1) => Ptr{Cuint},
             convert(Vector{Float32}, data.nzval) => Ptr{Cfloat},
             convert(UInt64, size(data.colptr)[1]) => Bst_ulong,
             convert(UInt64, nnz(data)) => Bst_ulong,
             out => Ref{DMatrixHandle})
    return out[]
end

function XGDMatrixCreateFromCSCT(data::SparseMatrixCSC)
    handle = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromCSR,
             convert(Vector{UInt64}, data.colptr - 1) => Ptr{Bst_ulong},
             convert(Vector{UInt32}, data.rowval - 1) => Ptr{Cuint},
             convert(Vector{Float32}, data.nzval) => Ptr{Cfloat},
             convert(UInt64, size(data.colptr)[1]) => Bst_ulong,
             convert(UInt64, nnz(data)) => Bst_ulong,
             handle => Ref{DMatrixHandle})
    return handle[]
end

function XGDMatrixCreateFromMat(data::Matrix{Float32}, missing::Float32)
    XGDMatrixCreateFromMatT(transpose(data), missing)
end

function XGDMatrixCreateFromMatT(data::Matrix{Float32}, missing::Float32)
    ncol, nrow = size(data)
    handle = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromMat,
             data => Ptr{Cfloat},
             nrow => Bst_ulong,
             ncol => Bst_ulong,
             missing => Cfloat,
             handle => Ref{DMatrixHandle})
    return handle[]
end

function XGDMatrixSliceDMatrix(handle::DMatrixHandle, idxset::Vector{Int32}, len::UInt64)
    ret = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixSliceDMatrix,
             handle => DMatrixHandle,
             idxset => Ptr{Cint},
             len => Bst_ulong,
             ret => Ref{DMatrixHandle})
    return ret[]
end

function XGDMatrixFree(handle::DMatrixHandle)
    @xgboost(:XGDMatrixFree,
             handle => DMatrixHandle)
end

function XGDMatrixSaveBinary(handle::DMatrixHandle, fname::String, silent::Int32)
    @xgboost(:XGDMatrixSaveBinary,
             handle => DMatrixHandle,
             fname => Cstring,
             silent => Cint)
end

function XGDMatrixSetFloatInfo(handle::DMatrixHandle, field::String, array::Vector{Float32},
                               len::UInt64)
    @xgboost(:XGDMatrixSetFloatInfo,
             handle => DMatrixHandle,
             field => Cstring,
             array => Ptr{Cfloat},
             len => Bst_ulong)
end

function XGDMatrixSetUIntInfo(handle::DMatrixHandle, field::String, array::Vector{UInt32},
                              len::UInt64)
    @xgboost(:XGDMatrixSetUIntInfo,
             handle => DMatrixHandle,
             field => Cstring,
             array => Ptr{Cuint},
             len => Bst_ulong)
end

function XGDMatrixSetGroup(handle::DMatrixHandle, array::Vector{UInt32}, len::UInt64)
    @xgboost(:XGDMatrixSetGroup,
             handle => DMatrixHandle,
             array => Ptr{Cuint},
             len => Bst_ulong)
end

function XGDMatrixGetFloatInfo(handle::DMatrixHandle, field::String, out_len::Vector{Bst_ulong})
    out_dptr = Ref{Ptr{Cfloat}}()
    @xgboost(:XGDMatrixGetFloatInfo,
             handle => DMatrixHandle,
             field => Cstring,
             out_len => Ptr{Bst_ulong},
             out_dptr =>  Ref{Ptr{Cfloat}})
    return out_dptr[]
end

function JLGetFloatInfo(handle::DMatrixHandle, field::String)
    len = Bst_ulong[1]
    ptr = XGDMatrixGetFloatInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function JLGetUintInfo(handle::DMatrixHandle, field::String)
    len = Bst_ulong[1]
    ptr = XGDMatrixGetUIntInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function XGDMatrixGetUIntInfo(handle::DMatrixHandle, field::String, out_len::Vector{Bst_ulong})
    out_dptr = Ref{Ptr{Cuint}}()
    @xgboost(:XGDMatrixGetUIntInfo,
             handle => DMatrixHandle,
             field => Cstring,
             out_len => Ptr{Bst_ulong},
             out_dptr => Ref{Ptr{Cuint}})
    return out_dptr[]
end

function XGDMatrixNumRow(handle::DMatrixHandle)
    out = Ref{Bst_ulong}()
    @xgboost(:XGDMatrixNumRow,
             handle => DMatrixHandle,
             out => Ref{Bst_ulong})
    return out[]
end

function XGBoosterCreate(cachelist::Vector{BoosterHandle}, len::Int64)
    out = Ref{BoosterHandle}()
    @xgboost(:XGBoosterCreate,
             cachelist => Ptr{BoosterHandle},
             len => Bst_ulong,
             out => Ref{BoosterHandle})
    return out[]
end

function XGBoosterFree(handle::BoosterHandle)
    @xgboost(:XGBoosterFree,
             handle => BoosterHandle)
end

function XGBoosterSetParam(handle::BoosterHandle, name::String, value::String)
    @xgboost(:XGBoosterSetParam,
             handle => BoosterHandle,
             name => Cstring,
             value => Cstring)
end

function XGBoosterUpdateOneIter(handle::BoosterHandle, iter::Int32, dtrain::DMatrixHandle)
    @xgboost(:XGBoosterUpdateOneIter,
             handle => BoosterHandle,
             iter => Cint,
             dtrain => DMatrixHandle)
end

function XGBoosterBoostOneIter(handle::BoosterHandle, dtrain::DMatrixHandle, grad::Vector{Float32},
                               hess::Vector{Float32}, len::UInt64)
    @xgboost(:XGBoosterBoostOneIter,
             handle => BoosterHandle,
             dtrain => DMatrixHandle,
             grad => Ptr{Cfloat},
             hess => Ptr{Cfloat},
             len => Bst_ulong)
end

function XGBoosterEvalOneIter(handle::BoosterHandle, iter::Int32, dmats::Vector{DMatrixHandle},
                              evnames::Vector{String}, len::UInt64)
    out_result = Ref{Cstring}()
    @xgboost(:XGBoosterEvalOneIter,
             handle => BoosterHandle,
             iter => Cint,
             dmats => Ptr{DMatrixHandle},
             evnames => Ptr{Cstring},
             len => Bst_ulong,
             out_result => Ref{Cstring})
    return unsafe_string(out_result[])
end

function XGBoosterPredict(handle::BoosterHandle, dmat::DMatrixHandle, option_mask::Int32,
                          ntree_limit::UInt32, out_len::Vector{UInt64})
    out_result = Ref{Ptr{Float32}}()
    @xgboost(:XGBoosterPredict,
             handle => BoosterHandle,
             dmat => DMatrixHandle,
             option_mask => Cint,
             ntree_limit => Cuint,
             out_len => Ptr{Bst_ulong},
             out_result => Ref{Ptr{Cfloat}})
    return out_result[]
end

function XGBoosterLoadModel(handle::BoosterHandle, fname::String)
    @xgboost(:XGBoosterLoadModel,
             handle => BoosterHandle,
             fname => Cstring)
end

function XGBoosterSaveModel(handle::BoosterHandle, fname::String)
    @xgboost(:XGBoosterSaveModel,
             handle => BoosterHandle,
             fname => Cstring)
end

function XGBoosterDumpModel(handle::BoosterHandle, fmap::String, with_stats::Int64)
    out_dump_array = Ref{Ptr{Cstring}}()
    out_len = Ref{Bst_ulong}(0)
    @xgboost(:XGBoosterDumpModel,
             handle => BoosterHandle,
             fmap => Cstring,
             with_stats => Cint,
             out_len => Ref{Bst_ulong},
             out_dump_array => Ref{Ptr{Cstring}})
    return unsafe_wrap(Array, out_dump_array[], out_len[])
end
