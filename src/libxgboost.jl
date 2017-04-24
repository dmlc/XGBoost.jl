include("../deps/deps.jl")


if build_version == "master"
    const Bst_ulong = Culonglong
else
    const Bst_ulong = Culong
end

const DMatrixHandle = Ptr{Void}
const BoosterHandle = Ptr{Void}


"Calls an xgboost API function and correctly reports errors."
macro xgboost(f, params...)
    return quote
        err = ccall(($f, _jl_libxgboost), Cint,
                    ($((esc(i.args[end]) for i in params)...),),
                    $((esc(i.args[end - 1]) for i in params)...))
        if err != 0
            err_msg = unsafe_string(ccall((:XGBGetLastError, _jl_libxgboost), Cstring, ()))
            error("Call to XGBoost C function ", string($(esc(f))), " failed: ", err_msg)
        end
    end
end


function XGDMatrixCreateFromFile(fname::String, silent::Bool)
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
             convert(Vector{Bst_ulong}, data.colptr - 1) => Ref{Bst_ulong},
             convert(Vector{Cuint}, data.rowval - 1) => Ref{Cuint},
             convert(Vector{Cfloat}, data.nzval) => Ref{Cfloat},
             length(data.colptr) => Bst_ulong,
             nnz(data) => Bst_ulong,
             out => Ref{DMatrixHandle})
    return out[]
end


function XGDMatrixCreateFromCSCT(data::SparseMatrixCSC)
    out = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromCSR,
             convert(Vector{Bst_ulong}, data.colptr - 1) => Ref{Bst_ulong},
             convert(Vector{Cuint}, data.rowval - 1) => Ref{Cuint},
             convert(Vector{Cfloat}, data.nzval) => Ref{Cfloat},
             length(data.colptr) => Bst_ulong,
             nnz(data) => Bst_ulong,
             out => Ref{DMatrixHandle})
    return out[]
end


function XGDMatrixCreateFromMat(data::Matrix{<:Real}, missing::Real)
    t_data = transpose(data)
    return XGDMatrixCreateFromMatT(t_data, missing)
end


function XGDMatrixCreateFromMatT(data::Matrix{Cfloat}, missing::Real)
    ncol, nrow = size(data)
    handle = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixCreateFromMat,
             data => Ref{Cfloat},
             nrow => Bst_ulong,
             ncol => Bst_ulong,
             missing => Cfloat,
             handle => Ref{DMatrixHandle})
    return handle[]
end


function XGDMatrixCreateFromMatT(data::Matrix{<:Real}, missing::Real)
    c_data = convert(Matrix{Cfloat}, data)
    return XGDMatrixCreateFromMatT(c_data, missing)
end


function XGDMatrixSliceDMatrix(handle::DMatrixHandle, idxset::Vector{Cint}, len::Integer)
    ret = Ref{DMatrixHandle}()
    @xgboost(:XGDMatrixSliceDMatrix,
             handle => DMatrixHandle,
             idxset => Ref{Cint},
             len => Bst_ulong,
             ret => Ref{DMatrixHandle})
    return ret[]
end


function XGDMatrixSliceDMatrix(handle::DMatrixHandle, idxset::Vector{<:Integer}, len::Integer)
    c_idxset = convert(Vector{Cint}, idxset)
    return XGDMatrixSliceDMatrix(handle, c_idxset, len)
end


function XGDMatrixFree(handle::DMatrixHandle)
    @xgboost(:XGDMatrixFree,
             handle => DMatrixHandle)
end


function XGDMatrixSaveBinary(handle::DMatrixHandle, fname::String, silent::Bool)
    @xgboost(:XGDMatrixSaveBinary,
             handle => DMatrixHandle,
             fname => Cstring,
             silent => Cint)
end


function XGDMatrixSetFloatInfo(handle::DMatrixHandle, field::String, array::Vector{Cfloat},
                               len::Integer)
    @xgboost(:XGDMatrixSetFloatInfo,
             handle => DMatrixHandle,
             field => Cstring,
             array => Ref{Cfloat},
             len => Bst_ulong)
end


function XGDMatrixSetFloatInfo(handle::DMatrixHandle, field::String, array::Vector{<:Real},
                               len::Integer)
    c_array = convert(Vector{Cfloat}, array)
    XGDMatrixSetFloatInfo(handle, field, c_array, len)
end


function XGDMatrixSetUIntInfo(handle::DMatrixHandle, field::String, array::Vector{Cuint},
                              len::Integer)
    @xgboost(:XGDMatrixSetUIntInfo,
             handle => DMatrixHandle,
             field => Cstring,
             array => Ref{Cuint},
             len => Bst_ulong)
end


function XGDMatrixSetUIntInfo(handle::DMatrixHandle, field::String, array::Vector{<:Integer},
                              len::Integer)
    c_array = convert(Vector{Cuint}, array)
    XGDMatrixSetUIntInfo(handle, field, c_array, len)
end


function XGDMatrixSetGroup(handle::DMatrixHandle, array::Vector{Cuint}, len::Integer)
    @xgboost(:XGDMatrixSetGroup,
             handle => DMatrixHandle,
             array => Ref{Cuint},
             len => Bst_ulong)
end


function XGDMatrixSetGroup(handle::DMatrixHandle, array::Vector{<:Integer}, len::Integer)
    c_array = convert(Vector{Cuint}, array)
    XGDMatrixSetGroup(handle, c_array, len)
end


function XGDMatrixGetFloatInfo(handle::DMatrixHandle, field::String)
    out_len = Ref{Bst_ulong}(0)
    out_dptr = Ref{Ptr{Cfloat}}()
    @xgboost(:XGDMatrixGetFloatInfo,
             handle => DMatrixHandle,
             field => Cstring,
             out_len => Ref{Bst_ulong},
             out_dptr =>  Ref{Ptr{Cfloat}})
    return unsafe_wrap(Array, out_dptr[], out_len[])
end


function XGDMatrixGetUIntInfo(handle::DMatrixHandle, field::String)
    out_len = Ref{Bst_ulong}(0)
    out_dptr = Ref{Ptr{Cuint}}()
    @xgboost(:XGDMatrixGetUIntInfo,
             handle => DMatrixHandle,
             field => Cstring,
             out_len => Ref{Bst_ulong},
             out_dptr => Ref{Ptr{Cuint}})
    return unsafe_wrap(Array, out_dptr[], out_len[])
end


function XGDMatrixNumRow(handle::DMatrixHandle)
    out = Ref{Bst_ulong}()
    @xgboost(:XGDMatrixNumRow,
             handle => DMatrixHandle,
             out => Ref{Bst_ulong})
    return out[]
end


function XGDMatrixNumCol(handle::DMatrixHandle)
    out = Ref{Bst_ulong}()
    @xgboost(:XGDMatrixNumCol,
             handle => DMatrixHandle,
             out => Ref{Bst_ulong})
    return out[]
end


function XGBoosterCreate(dmats::Vector{DMatrixHandle}, len::Integer)
    out = Ref{BoosterHandle}()
    @xgboost(:XGBoosterCreate,
             dmats => Ptr{DMatrixHandle},
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


function XGBoosterUpdateOneIter(handle::BoosterHandle, iter::Integer, dtrain::DMatrixHandle)
    @xgboost(:XGBoosterUpdateOneIter,
             handle => BoosterHandle,
             iter => Cint,
             dtrain => DMatrixHandle)
end


function XGBoosterBoostOneIter(handle::BoosterHandle, dtrain::DMatrixHandle, grad::Vector{Cfloat},
                               hess::Vector{Cfloat}, len::Integer)
    @xgboost(:XGBoosterBoostOneIter,
             handle => BoosterHandle,
             dtrain => DMatrixHandle,
             grad => Ref{Cfloat},
             hess => Ref{Cfloat},
             len => Bst_ulong)
end


function XGBoosterBoostOneIter(handle::BoosterHandle, dtrain::DMatrixHandle, grad::Vector{<:Real},
                               hess::Vector{<:Real}, len::Integer)
    c_grad = convert(Vector{Cfloat}, grad)
    c_hess = convert(Vector{Cfloat}, hess)
    XGBoosterBoostOneIter(handle, dtrain, c_grad, c_hess, len)
end


function XGBoosterEvalOneIter(handle::BoosterHandle, iter::Integer, dmats::Vector{DMatrixHandle},
                              evnames::Vector{String}, len::Integer)
    out_result = Ref{Cstring}()
    @xgboost(:XGBoosterEvalOneIter,
             handle => BoosterHandle,
             iter => Cint,
             dmats => Ref{DMatrixHandle},
             evnames => Ref{Cstring},
             len => Bst_ulong,
             out_result => Ref{Cstring})
    return unsafe_string(out_result[])
end


function XGBoosterPredict(handle::BoosterHandle, dmat::DMatrixHandle, option_mask::Integer,
                          ntree_limit::Integer)
    out_len = Ref{Bst_ulong}(0)
    out_result = Ref{Ptr{Cfloat}}()
    @xgboost(:XGBoosterPredict,
             handle => BoosterHandle,
             dmat => DMatrixHandle,
             option_mask => Cint,
             ntree_limit => Cuint,
             out_len => Ref{Bst_ulong},
             out_result => Ref{Ptr{Cfloat}})
    return deepcopy(unsafe_wrap(Array, out_result[], out_len[]))
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


function XGBoosterLoadModelFromBuffer(handle::BoosterHandle, buf::Vector{UInt8}, len::Integer)
    @xgboost(:XGBoosterLoadModelFromBuffer,
             handle => BoosterHandle,
             buf => Ref{UInt8},
             len => Bst_ulong)
end


function XGBoosterGetModelRaw(handle::BoosterHandle)
    out_len = Ref{Bst_ulong}()
    out_dptr = Ref{Ptr{UInt8}}()
    @xgboost(:XGBoosterGetModelRaw,
             handle => BoosterHandle,
             out_len => Ref{Bst_ulong},
             out_dptr => Ref{Ptr{UInt8}})
    return deepcopy(unsafe_wrap(Array, out_dptr[], out_len[]))
end


function XGBoosterDumpModel(handle::BoosterHandle, fmap::String, with_stats::Integer)
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


function XGBoosterGetAttr(handle::BoosterHandle, key::String)
    out = Ref{Cstring}()
    success = Ref{Cint}(-1)
    @xgboost(:XGBoosterGetAttr,
             handle => BoosterHandle,
             key => Cstring,
             out => Ref{Cstring},
             success => Ref{Cint})
    return success[] == 1 ? unsafe_string(out[]) : ""
end


function XGBoosterSetAttr(handle::BoosterHandle, key::String, value::String)
    @xgboost(:XGBoosterSetAttr,
             handle => BoosterHandle,
             key => Cstring,
             value => Cstring)
end


function XGBoosterSetAttr(handle::BoosterHandle, key::String, value::Ptr{Void})
    @xgboost(:XGBoosterSetAttr,
             handle => BoosterHandle,
             key => Cstring,
             value => Ptr{Void})
end


function XGBoosterGetAttrNames(handle::BoosterHandle)
    out_len = Ref{Bst_ulong}(0)
    out = Ref{Ptr{Cstring}}()
    @xgboost(:XGBoosterGetAttrNames,
             handle => BoosterHandle,
             out_len => Ref{Bst_ulong},
             out => Ref{Ptr{Cstring}})
    out_ptrs = unsafe_wrap(Array, out[], out_len[])
    return [unsafe_string(ptr) for ptr in out_ptrs]
end
