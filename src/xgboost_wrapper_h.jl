const Bst_ulong = Culonglong

"Calls an xgboost API function and correctly reports errors."
macro xgboost(f, params...)
    return quote
        err = ccall(($f, libxgboost), Int64,
                    ($((esc(i.args[end]) for i in params)...),),
                    $((esc(i.args[end-1]) for i in params)...))
        if err != 0
            err_msg = unsafe_string(ccall((:XGBGetLastError, libxgboost), Cstring, ()))
            error("Call to XGBoost C function ", string($(esc(f))), " failed: ", err_msg)
        end
    end
end

function XGDMatrixCreateFromFile(fname::String, silent::Int32)
    out = Ref{Ptr{Nothing}}()
    @xgboost(:XGDMatrixCreateFromFile,
             fname => Cstring,
             silent => Cint,
             out => Ref{Ptr{Nothing}})
    return out[]
end

@deprecate XGDMatrixCreateFromCSC(data) XGDMatrixCreateFromCSCEx(data) false

function XGDMatrixCreateFromCSCEx(data::SparseMatrixCSC)
    out = Ref{Ptr{Nothing}}()
    @xgboost(:XGDMatrixCreateFromCSCEx,
             convert(Vector{UInt64}, data.colptr .- 1) => Ptr{Csize_t},
             convert(Vector{UInt32}, data.rowval .- 1) => Ptr{Cuint},
             convert(Vector{Float32}, data.nzval) => Ptr{Cfloat},
             convert(UInt64, size(data.colptr)[1]) => Csize_t,
             convert(UInt64, nnz(data)) => Csize_t,
             convert(UInt64, size(data)[1]) => Csize_t, #num_rows
             out => Ref{Ptr{Nothing}})
    return out[]
end

function XGDMatrixCreateFromCSCT(data::SparseMatrixCSC)
    handle = Ref{Ptr{Nothing}}()
    @xgboost(:XGDMatrixCreateFromCSREx,
             convert(Vector{UInt64}, data.colptr .- 1) => Ptr{Csize_t},
             convert(Vector{UInt32}, data.rowval .- 1) => Ptr{Cuint},
             convert(Vector{Float32}, data.nzval) => Ptr{Cfloat},
             convert(UInt64, size(data.colptr)[1]) => Csize_t,
             convert(UInt64, nnz(data)) => Csize_t,
             convert(UInt64, size(data)[2]) => Csize_t, #num_cols
             handle => Ref{Ptr{Nothing}})
    return handle[]
end

function XGDMatrixCreateFromMat(data::Matrix{Float32}, missing::Float32)
    XGDMatrixCreateFromMatT(Matrix(transpose(data)), missing)
end

function XGDMatrixCreateFromMatT(data::Matrix{Float32}, missing::Float32)
    ncol, nrow = size(data)
    handle = Ref{Ptr{Nothing}}()
    @xgboost(:XGDMatrixCreateFromMat,
             data => Ptr{Cfloat},
             nrow => Bst_ulong,
             ncol => Bst_ulong,
             missing => Cfloat,
             handle => Ref{Ptr{Nothing}})
    return handle[]
end

function XGDMatrixSliceDMatrix(handle::Ptr{Nothing}, idxset::Vector{Int32}, len::UInt64)
    ret = Ref{Ptr{Nothing}}()
    @xgboost(:XGDMatrixSliceDMatrix,
             handle => Ptr{Nothing},
             idxset => Ptr{Cint},
             len => Bst_ulong,
             ret => Ref{Ptr{Nothing}})
    return ret[]
end

function XGDMatrixFree(handle::Ptr{Nothing})
    @xgboost(:XGDMatrixFree,
             handle => Ptr{Nothing})
end

function XGDMatrixSaveBinary(handle::Ptr{Nothing}, fname::String, silent::Int32)
    @xgboost(:XGDMatrixSaveBinary,
             handle => Ptr{Nothing},
             fname => Cstring,
             silent => Cint)
end

function XGDMatrixSetFloatInfo(handle::Ptr{Nothing}, field::String, array::Vector{Float32},
                               len::UInt64)
    @xgboost(:XGDMatrixSetFloatInfo,
             handle => Ptr{Nothing},
             field => Cstring,
             array => Ptr{Cfloat},
             len => Bst_ulong)
end

function XGDMatrixSetUIntInfo(handle::Ptr{Nothing}, field::String, array::Vector{UInt32},
                              len::UInt64)
    @xgboost(:XGDMatrixSetUIntInfo,
             handle => Ptr{Nothing},
             field => Cstring,
             array => Ptr{Cuint},
             len => Bst_ulong)
end

function XGDMatrixSetGroup(handle::Ptr{Nothing}, array::Vector{UInt32}, len::UInt64)
    @xgboost(:XGDMatrixSetGroup,
             handle => Ptr{Nothing},
             array => Ptr{Cuint},
             len => Bst_ulong)
end

function XGDMatrixGetFloatInfo(handle::Ptr{Nothing}, field::String, out_len::Vector{Bst_ulong})
    out_dptr = Ref{Ptr{Cfloat}}()
    @xgboost(:XGDMatrixGetFloatInfo,
             handle => Ptr{Nothing},
             field => Cstring,
             out_len => Ptr{Bst_ulong},
             out_dptr =>  Ref{Ptr{Cfloat}})
    return out_dptr[]
end

function JLGetFloatInfo(handle::Ptr{Nothing}, field::String)
    len = Bst_ulong[1]
    ptr = XGDMatrixGetFloatInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function JLGetUintInfo(handle::Ptr{Nothing}, field::String)
    len = Bst_ulong[1]
    ptr = XGDMatrixGetUIntInfo(handle, field, len)
    return unsafe_wrap(Array, ptr, len[1])
end

function XGDMatrixGetUIntInfo(handle::Ptr{Nothing}, field::String, out_len::Vector{Bst_ulong})
    out_dptr = Ref{Ptr{Cuint}}()
    @xgboost(:XGDMatrixGetUIntInfo,
             handle => Ptr{Nothing},
             field => Cstring,
             out_len => Ptr{Bst_ulong},
             out_dptr => Ref{Ptr{Cuint}})
    return out_dptr[]
end

function XGDMatrixNumRow(handle::Ptr{Nothing})
    out = Ref{Bst_ulong}()
    @xgboost(:XGDMatrixNumRow,
             handle => Ptr{Nothing},
             out => Ref{Bst_ulong})
    return out[]
end

function XGBoosterCreate(cachelist::Vector{Ptr{Nothing}}, len::Int64)
    out = Ref{Ptr{Nothing}}()
    @xgboost(:XGBoosterCreate,
             cachelist => Ptr{Ptr{Nothing}},
             len => Bst_ulong,
             out => Ref{Ptr{Nothing}})
    return out[]
end

function XGBoosterFree(handle::Ptr{Nothing})
    @xgboost(:XGBoosterFree,
             handle => Ptr{Nothing})
end

function XGBoosterSetParam(handle::Ptr{Nothing}, name::String, value::String)
    @xgboost(:XGBoosterSetParam,
             handle => Ptr{Nothing},
             name => Cstring,
             value => Cstring)
end

function XGBoosterUpdateOneIter(handle::Ptr{Nothing}, iter::Int32, dtrain::Ptr{Nothing})
    @xgboost(:XGBoosterUpdateOneIter,
             handle => Ptr{Nothing},
             iter => Cint,
             dtrain => Ptr{Nothing})
end

function XGBoosterBoostOneIter(handle::Ptr{Nothing}, dtrain::Ptr{Nothing}, grad::Vector{Float32},
                               hess::Vector{Float32}, len::UInt64)
    @xgboost(:XGBoosterBoostOneIter,
             handle => Ptr{Nothing},
             dtrain => Ptr{Nothing},
             grad => Ptr{Cfloat},
             hess => Ptr{Cfloat},
             len => Bst_ulong)
end

function XGBoosterEvalOneIter(handle::Ptr{Nothing}, iter::Int32, dmats::Vector{Ptr{Nothing}},
                              evnames::Vector{String}, len::UInt64)
    out_result = Ref{Cstring}()
    @xgboost(:XGBoosterEvalOneIter,
             handle => Ptr{Nothing},
             iter => Cint,
             dmats => Ptr{Ptr{Nothing}},
             evnames => Ptr{Cstring},
             len => Bst_ulong,
             out_result => Ref{Cstring})
    return unsafe_string(out_result[])
end

function XGBoosterPredict(handle::Ptr{Nothing}, dmat::Ptr{Nothing}, option_mask::Int32,
                          ntree_limit::UInt32, training::Int32, out_len::Vector{UInt64})
    out_result = Ref{Ptr{Float32}}()
    @xgboost(:XGBoosterPredict,
             handle => Ptr{Nothing},
             dmat => Ptr{Nothing},
             option_mask => Cint,
             ntree_limit => Cuint,
             training => Cint, 
             out_len => Ptr{Bst_ulong},
             out_result => Ref{Ptr{Cfloat}})
    return out_result[]
end

function XGBoosterLoadModel(handle::Ptr{Nothing}, fname::String)
    @xgboost(:XGBoosterLoadModel,
             handle => Ptr{Nothing},
             fname => Cstring)
end

function XGBoosterSaveModel(handle::Ptr{Nothing}, fname::String)
    @xgboost(:XGBoosterSaveModel,
             handle => Ptr{Nothing},
             fname => Cstring)
end


function XGBoosterDumpModel(handle::Ptr{Nothing}, fmap::String, with_stats::Int64, dump_format::String="text")
    out_dump_array = Ref{Ptr{Cstring}}()
    out_len = Ref{Bst_ulong}(0)
    @xgboost(:XGBoosterDumpModelEx,
             handle => Ptr{Nothing},
             fmap => Cstring,
             with_stats => Cint,
             dump_format => Cstring,
             out_len => Ref{Bst_ulong},
             out_dump_array => Ref{Ptr{Cstring}})
    return unsafe_wrap(Array, out_dump_array[], out_len[])
end
