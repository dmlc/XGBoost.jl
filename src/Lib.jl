module Lib

using XGBoost_jll
using XGBoost_GPU_jll

using CEnum

# only enable GPU support if there is a valid binary compatible with this system
# should we place a warning here?
if XGBoost_GPU_jll.is_available()
    lib_xgboost = XGBoost_GPU_jll.libxgboost
else
    lib_xgboost = XGBoost_jll.libxgboost
end

struct XGBoostError <: Exception
    caller
    message::String
end

function Base.showerror(io::IO, err::XGBoostError)
    println(io, "XGBoostError: (caller: $(string(err.caller)))")
    print(io, err.message)
end

"""
    xgbcall(𝒻, a...)

Call xgboost library function `𝒻` on arguments `a` properly handling errors.
"""
function xgbcall(𝒻, a...)
    err = 𝒻(a...)
    if err ≠ 0
        msg = unsafe_string(XGBGetLastError())
        throw(XGBoostError(𝒻, msg))
    end
    err
end

export XGBoostError, xgbcall


const bst_ulong = UInt64

const DMatrixHandle = Ptr{Cvoid}

const BoosterHandle = Ptr{Cvoid}

function XGBoostVersion(major, minor, patch)
    @ccall lib_xgboost.XGBoostVersion(major::Ptr{Cint}, minor::Ptr{Cint}, patch::Ptr{Cint})::Cvoid
end

function XGBuildInfo(out)
    @ccall lib_xgboost.XGBuildInfo(out::Ptr{Ptr{Cchar}})::Cint
end

# no prototype is found for this function at c_api.h:84:21, please use with caution
function XGBGetLastError()
    @ccall lib_xgboost.XGBGetLastError()::Ptr{Cchar}
end

function XGBRegisterLogCallback(callback)
    @ccall lib_xgboost.XGBRegisterLogCallback(callback::Ptr{Cvoid})::Cint
end

function XGBSetGlobalConfig(config)
    @ccall lib_xgboost.XGBSetGlobalConfig(config::Ptr{Cchar})::Cint
end

function XGBGetGlobalConfig(out_config)
    @ccall lib_xgboost.XGBGetGlobalConfig(out_config::Ptr{Ptr{Cchar}})::Cint
end

function XGDMatrixCreateFromFile(fname, silent, out)
    @ccall lib_xgboost.XGDMatrixCreateFromFile(fname::Ptr{Cchar}, silent::Cint, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromURI(config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromURI(config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromCSREx(indptr, indices, data, nindptr, nelem, num_col, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCSREx(indptr::Ptr{Csize_t}, indices::Ptr{Cuint}, data::Ptr{Cfloat}, nindptr::Csize_t, nelem::Csize_t, num_col::Csize_t, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromCSR(indptr, indices, data, ncol, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCSR(indptr::Ptr{Cchar}, indices::Ptr{Cchar}, data::Ptr{Cchar}, ncol::bst_ulong, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromDense(data, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromDense(data::Ptr{Cchar}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromCSC(indptr, indices, data, nrow, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCSC(indptr::Ptr{Cchar}, indices::Ptr{Cchar}, data::Ptr{Cchar}, nrow::bst_ulong, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromCSCEx(col_ptr, indices, data, nindptr, nelem, num_row, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCSCEx(col_ptr::Ptr{Csize_t}, indices::Ptr{Cuint}, data::Ptr{Cfloat}, nindptr::Csize_t, nelem::Csize_t, num_row::Csize_t, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromMat(data, nrow, ncol, missing, out)
    @ccall lib_xgboost.XGDMatrixCreateFromMat(data::Ptr{Cfloat}, nrow::bst_ulong, ncol::bst_ulong, missing::Cfloat, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromMat_omp(data, nrow, ncol, missing, out, nthread)
    @ccall lib_xgboost.XGDMatrixCreateFromMat_omp(data::Ptr{Cfloat}, nrow::bst_ulong, ncol::bst_ulong, missing::Cfloat, out::Ptr{DMatrixHandle}, nthread::Cint)::Cint
end

function XGDMatrixCreateFromDT(data, feature_stypes, nrow, ncol, out, nthread)
    @ccall lib_xgboost.XGDMatrixCreateFromDT(data::Ptr{Ptr{Cvoid}}, feature_stypes::Ptr{Ptr{Cchar}}, nrow::bst_ulong, ncol::bst_ulong, out::Ptr{DMatrixHandle}, nthread::Cint)::Cint
end

function XGDMatrixCreateFromCudaColumnar(data, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCudaColumnar(data::Ptr{Cchar}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixCreateFromCudaArrayInterface(data, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCudaArrayInterface(data::Ptr{Cchar}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

const DataIterHandle = Ptr{Cvoid}

const DataHolderHandle = Ptr{Cvoid}

struct XGBoostBatchCSR
    size::Csize_t
    columns::Csize_t
    offset::Ptr{Int64}
    label::Ptr{Cfloat}
    weight::Ptr{Cfloat}
    index::Ptr{Cint}
    value::Ptr{Cfloat}
end

# typedef int XGBCallbackSetData ( // NOLINT(*) DataHolderHandle handle , XGBoostBatchCSR batch )
const XGBCallbackSetData = Cvoid

# typedef int XGBCallbackDataIterNext ( // NOLINT(*) DataIterHandle data_handle , XGBCallbackSetData * set_function , DataHolderHandle set_function_handle )
const XGBCallbackDataIterNext = Cvoid

function XGDMatrixCreateFromDataIter(data_handle, callback, cache_info, out)
    @ccall lib_xgboost.XGDMatrixCreateFromDataIter(data_handle::DataIterHandle, callback::Ptr{XGBCallbackDataIterNext}, cache_info::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGProxyDMatrixCreate(out)
    @ccall lib_xgboost.XGProxyDMatrixCreate(out::Ptr{DMatrixHandle})::Cint
end

# typedef int XGDMatrixCallbackNext ( DataIterHandle iter )
const XGDMatrixCallbackNext = Cvoid

# typedef void DataIterResetCallback ( DataIterHandle handle )
const DataIterResetCallback = Cvoid

function XGDMatrixCreateFromCallback(iter, proxy, reset, next, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromCallback(iter::DataIterHandle, proxy::DMatrixHandle, reset::Ptr{DataIterResetCallback}, next::Ptr{XGDMatrixCallbackNext}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGQuantileDMatrixCreateFromCallback(iter, proxy, ref, reset, next, config, out)
    @ccall lib_xgboost.XGQuantileDMatrixCreateFromCallback(iter::DataIterHandle, proxy::DMatrixHandle, ref::DataIterHandle, reset::Ptr{DataIterResetCallback}, next::Ptr{XGDMatrixCallbackNext}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDeviceQuantileDMatrixCreateFromCallback(iter, proxy, reset, next, missing, nthread, max_bin, out)
    @ccall lib_xgboost.XGDeviceQuantileDMatrixCreateFromCallback(iter::DataIterHandle, proxy::DMatrixHandle, reset::Ptr{DataIterResetCallback}, next::Ptr{XGDMatrixCallbackNext}, missing::Cfloat, nthread::Cint, max_bin::Cint, out::Ptr{DMatrixHandle})::Cint
end

function XGProxyDMatrixSetDataCudaArrayInterface(handle, c_interface_str)
    @ccall lib_xgboost.XGProxyDMatrixSetDataCudaArrayInterface(handle::DMatrixHandle, c_interface_str::Ptr{Cchar})::Cint
end

function XGProxyDMatrixSetDataCudaColumnar(handle, c_interface_str)
    @ccall lib_xgboost.XGProxyDMatrixSetDataCudaColumnar(handle::DMatrixHandle, c_interface_str::Ptr{Cchar})::Cint
end

function XGProxyDMatrixSetDataDense(handle, c_interface_str)
    @ccall lib_xgboost.XGProxyDMatrixSetDataDense(handle::DMatrixHandle, c_interface_str::Ptr{Cchar})::Cint
end

function XGProxyDMatrixSetDataCSR(handle, indptr, indices, data, ncol)
    @ccall lib_xgboost.XGProxyDMatrixSetDataCSR(handle::DMatrixHandle, indptr::Ptr{Cchar}, indices::Ptr{Cchar}, data::Ptr{Cchar}, ncol::bst_ulong)::Cint
end

function XGImportArrowRecordBatch(data_handle, ptr_array, ptr_schema)
    @ccall lib_xgboost.XGImportArrowRecordBatch(data_handle::DataIterHandle, ptr_array::Ptr{Cvoid}, ptr_schema::Ptr{Cvoid})::Cint
end

function XGDMatrixCreateFromArrowCallback(next, config, out)
    @ccall lib_xgboost.XGDMatrixCreateFromArrowCallback(next::Ptr{XGDMatrixCallbackNext}, config::Ptr{Cchar}, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixSliceDMatrix(handle, idxset, len, out)
    @ccall lib_xgboost.XGDMatrixSliceDMatrix(handle::DMatrixHandle, idxset::Ptr{Cint}, len::bst_ulong, out::Ptr{DMatrixHandle})::Cint
end

function XGDMatrixSliceDMatrixEx(handle, idxset, len, out, allow_groups)
    @ccall lib_xgboost.XGDMatrixSliceDMatrixEx(handle::DMatrixHandle, idxset::Ptr{Cint}, len::bst_ulong, out::Ptr{DMatrixHandle}, allow_groups::Cint)::Cint
end

function XGDMatrixFree(handle)
    @ccall lib_xgboost.XGDMatrixFree(handle::DMatrixHandle)::Cint
end

function XGDMatrixSaveBinary(handle, fname, silent)
    @ccall lib_xgboost.XGDMatrixSaveBinary(handle::DMatrixHandle, fname::Ptr{Cchar}, silent::Cint)::Cint
end

function XGDMatrixSetInfoFromInterface(handle, field, c_interface_str)
    @ccall lib_xgboost.XGDMatrixSetInfoFromInterface(handle::DMatrixHandle, field::Ptr{Cchar}, c_interface_str::Ptr{Cchar})::Cint
end

function XGDMatrixSetFloatInfo(handle, field, array, len)
    @ccall lib_xgboost.XGDMatrixSetFloatInfo(handle::DMatrixHandle, field::Ptr{Cchar}, array::Ptr{Cfloat}, len::bst_ulong)::Cint
end

function XGDMatrixSetUIntInfo(handle, field, array, len)
    @ccall lib_xgboost.XGDMatrixSetUIntInfo(handle::DMatrixHandle, field::Ptr{Cchar}, array::Ptr{Cuint}, len::bst_ulong)::Cint
end

function XGDMatrixSetStrFeatureInfo(handle, field, features, size)
    @ccall lib_xgboost.XGDMatrixSetStrFeatureInfo(handle::DMatrixHandle, field::Ptr{Cchar}, features::Ptr{Ptr{Cchar}}, size::bst_ulong)::Cint
end

function XGDMatrixGetStrFeatureInfo(handle, field, size, out_features)
    @ccall lib_xgboost.XGDMatrixGetStrFeatureInfo(handle::DMatrixHandle, field::Ptr{Cchar}, size::Ptr{bst_ulong}, out_features::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGDMatrixSetDenseInfo(handle, field, data, size, type)
    @ccall lib_xgboost.XGDMatrixSetDenseInfo(handle::DMatrixHandle, field::Ptr{Cchar}, data::Ptr{Cvoid}, size::bst_ulong, type::Cint)::Cint
end

function XGDMatrixSetGroup(handle, group, len)
    @ccall lib_xgboost.XGDMatrixSetGroup(handle::DMatrixHandle, group::Ptr{Cuint}, len::bst_ulong)::Cint
end

function XGDMatrixGetFloatInfo(handle, field, out_len, out_dptr)
    @ccall lib_xgboost.XGDMatrixGetFloatInfo(handle::DMatrixHandle, field::Ptr{Cchar}, out_len::Ptr{bst_ulong}, out_dptr::Ptr{Ptr{Cfloat}})::Cint
end

function XGDMatrixGetUIntInfo(handle, field, out_len, out_dptr)
    @ccall lib_xgboost.XGDMatrixGetUIntInfo(handle::DMatrixHandle, field::Ptr{Cchar}, out_len::Ptr{bst_ulong}, out_dptr::Ptr{Ptr{Cuint}})::Cint
end

function XGDMatrixNumRow(handle, out)
    @ccall lib_xgboost.XGDMatrixNumRow(handle::DMatrixHandle, out::Ptr{bst_ulong})::Cint
end

function XGDMatrixNumCol(handle, out)
    @ccall lib_xgboost.XGDMatrixNumCol(handle::DMatrixHandle, out::Ptr{bst_ulong})::Cint
end

function XGDMatrixNumNonMissing(handle, out)
    @ccall lib_xgboost.XGDMatrixNumNonMissing(handle::DMatrixHandle, out::Ptr{bst_ulong})::Cint
end

function XGDMatrixGetDataAsCSR(handle, config, out_indptr, out_indices, out_data)
    @ccall lib_xgboost.XGDMatrixGetDataAsCSR(handle::DMatrixHandle, config::Ptr{Cchar}, out_indptr::Ptr{bst_ulong}, out_indices::Ptr{Cuint}, out_data::Ptr{Cfloat})::Cint
end

function XGDMatrixGetQuantileCut(handle, config, out_indptr, out_data)
    @ccall lib_xgboost.XGDMatrixGetQuantileCut(handle::DMatrixHandle, config::Ptr{Cchar}, out_indptr::Ptr{Ptr{Cchar}}, out_data::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterCreate(dmats, len, out)
    @ccall lib_xgboost.XGBoosterCreate(dmats::Ptr{DMatrixHandle}, len::bst_ulong, out::Ptr{BoosterHandle})::Cint
end

function XGBoosterFree(handle)
    @ccall lib_xgboost.XGBoosterFree(handle::BoosterHandle)::Cint
end

function XGBoosterSlice(handle, begin_layer, end_layer, step, out)
    @ccall lib_xgboost.XGBoosterSlice(handle::BoosterHandle, begin_layer::Cint, end_layer::Cint, step::Cint, out::Ptr{BoosterHandle})::Cint
end

function XGBoosterBoostedRounds(handle, out)
    @ccall lib_xgboost.XGBoosterBoostedRounds(handle::BoosterHandle, out::Ptr{Cint})::Cint
end

function XGBoosterSetParam(handle, name, value)
    @ccall lib_xgboost.XGBoosterSetParam(handle::BoosterHandle, name::Ptr{Cchar}, value::Ptr{Cchar})::Cint
end

function XGBoosterGetNumFeature(handle, out)
    @ccall lib_xgboost.XGBoosterGetNumFeature(handle::BoosterHandle, out::Ptr{bst_ulong})::Cint
end

function XGBoosterUpdateOneIter(handle, iter, dtrain)
    @ccall lib_xgboost.XGBoosterUpdateOneIter(handle::BoosterHandle, iter::Cint, dtrain::DMatrixHandle)::Cint
end

function XGBoosterBoostOneIter(handle, dtrain, grad, hess, len)
    @ccall lib_xgboost.XGBoosterBoostOneIter(handle::BoosterHandle, dtrain::DMatrixHandle, grad::Ptr{Cfloat}, hess::Ptr{Cfloat}, len::bst_ulong)::Cint
end

function XGBoosterEvalOneIter(handle, iter, dmats, evnames, len, out_result)
    @ccall lib_xgboost.XGBoosterEvalOneIter(handle::BoosterHandle, iter::Cint, dmats::Ptr{DMatrixHandle}, evnames::Ptr{Ptr{Cchar}}, len::bst_ulong, out_result::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterPredict(handle, dmat, option_mask, ntree_limit, training, out_len, out_result)
    @ccall lib_xgboost.XGBoosterPredict(handle::BoosterHandle, dmat::DMatrixHandle, option_mask::Cint, ntree_limit::Cuint, training::Cint, out_len::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterPredictFromDMatrix(handle, dmat, config, out_shape, out_dim, out_result)
    @ccall lib_xgboost.XGBoosterPredictFromDMatrix(handle::BoosterHandle, dmat::DMatrixHandle, config::Ptr{Cchar}, out_shape::Ptr{Ptr{bst_ulong}}, out_dim::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterPredictFromDense(handle, values, config, m, out_shape, out_dim, out_result)
    @ccall lib_xgboost.XGBoosterPredictFromDense(handle::BoosterHandle, values::Ptr{Cchar}, config::Ptr{Cchar}, m::DMatrixHandle, out_shape::Ptr{Ptr{bst_ulong}}, out_dim::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterPredictFromCSR(handle, indptr, indices, values, ncol, config, m, out_shape, out_dim, out_result)
    @ccall lib_xgboost.XGBoosterPredictFromCSR(handle::BoosterHandle, indptr::Ptr{Cchar}, indices::Ptr{Cchar}, values::Ptr{Cchar}, ncol::bst_ulong, config::Ptr{Cchar}, m::DMatrixHandle, out_shape::Ptr{Ptr{bst_ulong}}, out_dim::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterPredictFromCudaArray(handle, values, config, m, out_shape, out_dim, out_result)
    @ccall lib_xgboost.XGBoosterPredictFromCudaArray(handle::BoosterHandle, values::Ptr{Cchar}, config::Ptr{Cchar}, m::DMatrixHandle, out_shape::Ptr{Ptr{bst_ulong}}, out_dim::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterPredictFromCudaColumnar(handle, values, config, m, out_shape, out_dim, out_result)
    @ccall lib_xgboost.XGBoosterPredictFromCudaColumnar(handle::BoosterHandle, values::Ptr{Cchar}, config::Ptr{Cchar}, m::DMatrixHandle, out_shape::Ptr{Ptr{bst_ulong}}, out_dim::Ptr{bst_ulong}, out_result::Ptr{Ptr{Cfloat}})::Cint
end

function XGBoosterLoadModel(handle, fname)
    @ccall lib_xgboost.XGBoosterLoadModel(handle::BoosterHandle, fname::Ptr{Cchar})::Cint
end

function XGBoosterSaveModel(handle, fname)
    @ccall lib_xgboost.XGBoosterSaveModel(handle::BoosterHandle, fname::Ptr{Cchar})::Cint
end

function XGBoosterLoadModelFromBuffer(handle, buf, len)
    @ccall lib_xgboost.XGBoosterLoadModelFromBuffer(handle::BoosterHandle, buf::Ptr{Cvoid}, len::bst_ulong)::Cint
end

function XGBoosterSaveModelToBuffer(handle, config, out_len, out_dptr)
    @ccall lib_xgboost.XGBoosterSaveModelToBuffer(handle::BoosterHandle, config::Ptr{Cchar}, out_len::Ptr{bst_ulong}, out_dptr::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterGetModelRaw(handle, out_len, out_dptr)
    @ccall lib_xgboost.XGBoosterGetModelRaw(handle::BoosterHandle, out_len::Ptr{bst_ulong}, out_dptr::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterSerializeToBuffer(handle, out_len, out_dptr)
    @ccall lib_xgboost.XGBoosterSerializeToBuffer(handle::BoosterHandle, out_len::Ptr{bst_ulong}, out_dptr::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterUnserializeFromBuffer(handle, buf, len)
    @ccall lib_xgboost.XGBoosterUnserializeFromBuffer(handle::BoosterHandle, buf::Ptr{Cvoid}, len::bst_ulong)::Cint
end

function XGBoosterLoadRabitCheckpoint(handle, version)
    @ccall lib_xgboost.XGBoosterLoadRabitCheckpoint(handle::BoosterHandle, version::Ptr{Cint})::Cint
end

function XGBoosterSaveRabitCheckpoint(handle)
    @ccall lib_xgboost.XGBoosterSaveRabitCheckpoint(handle::BoosterHandle)::Cint
end

function XGBoosterSaveJsonConfig(handle, out_len, out_str)
    @ccall lib_xgboost.XGBoosterSaveJsonConfig(handle::BoosterHandle, out_len::Ptr{bst_ulong}, out_str::Ptr{Ptr{Cchar}})::Cint
end

function XGBoosterLoadJsonConfig(handle, config)
    @ccall lib_xgboost.XGBoosterLoadJsonConfig(handle::BoosterHandle, config::Ptr{Cchar})::Cint
end

function XGBoosterDumpModel(handle, fmap, with_stats, out_len, out_dump_array)
    @ccall lib_xgboost.XGBoosterDumpModel(handle::BoosterHandle, fmap::Ptr{Cchar}, with_stats::Cint, out_len::Ptr{bst_ulong}, out_dump_array::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterDumpModelEx(handle, fmap, with_stats, format, out_len, out_dump_array)
    @ccall lib_xgboost.XGBoosterDumpModelEx(handle::BoosterHandle, fmap::Ptr{Cchar}, with_stats::Cint, format::Ptr{Cchar}, out_len::Ptr{bst_ulong}, out_dump_array::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterDumpModelWithFeatures(handle, fnum, fname, ftype, with_stats, out_len, out_models)
    @ccall lib_xgboost.XGBoosterDumpModelWithFeatures(handle::BoosterHandle, fnum::Cint, fname::Ptr{Ptr{Cchar}}, ftype::Ptr{Ptr{Cchar}}, with_stats::Cint, out_len::Ptr{bst_ulong}, out_models::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterDumpModelExWithFeatures(handle, fnum, fname, ftype, with_stats, format, out_len, out_models)
    @ccall lib_xgboost.XGBoosterDumpModelExWithFeatures(handle::BoosterHandle, fnum::Cint, fname::Ptr{Ptr{Cchar}}, ftype::Ptr{Ptr{Cchar}}, with_stats::Cint, format::Ptr{Cchar}, out_len::Ptr{bst_ulong}, out_models::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterGetAttr(handle, key, out, success)
    @ccall lib_xgboost.XGBoosterGetAttr(handle::BoosterHandle, key::Ptr{Cchar}, out::Ptr{Ptr{Cchar}}, success::Ptr{Cint})::Cint
end

function XGBoosterSetAttr(handle, key, value)
    @ccall lib_xgboost.XGBoosterSetAttr(handle::BoosterHandle, key::Ptr{Cchar}, value::Ptr{Cchar})::Cint
end

function XGBoosterGetAttrNames(handle, out_len, out)
    @ccall lib_xgboost.XGBoosterGetAttrNames(handle::BoosterHandle, out_len::Ptr{bst_ulong}, out::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterSetStrFeatureInfo(handle, field, features, size)
    @ccall lib_xgboost.XGBoosterSetStrFeatureInfo(handle::BoosterHandle, field::Ptr{Cchar}, features::Ptr{Ptr{Cchar}}, size::bst_ulong)::Cint
end

function XGBoosterGetStrFeatureInfo(handle, field, len, out_features)
    @ccall lib_xgboost.XGBoosterGetStrFeatureInfo(handle::BoosterHandle, field::Ptr{Cchar}, len::Ptr{bst_ulong}, out_features::Ptr{Ptr{Ptr{Cchar}}})::Cint
end

function XGBoosterFeatureScore(handle, config, out_n_features, out_features, out_dim, out_shape, out_scores)
    @ccall lib_xgboost.XGBoosterFeatureScore(handle::BoosterHandle, config::Ptr{Cchar}, out_n_features::Ptr{bst_ulong}, out_features::Ptr{Ptr{Ptr{Cchar}}}, out_dim::Ptr{bst_ulong}, out_shape::Ptr{Ptr{bst_ulong}}, out_scores::Ptr{Ptr{Cfloat}})::Cint
end

function XGCommunicatorInit(config)
    @ccall lib_xgboost.XGCommunicatorInit(config::Ptr{Cchar})::Cint
end

function XGCommunicatorFinalize()
    @ccall lib_xgboost.XGCommunicatorFinalize()::Cint
end

function XGCommunicatorGetRank()
    @ccall lib_xgboost.XGCommunicatorGetRank()::Cint
end

function XGCommunicatorGetWorldSize()
    @ccall lib_xgboost.XGCommunicatorGetWorldSize()::Cint
end

function XGCommunicatorIsDistributed()
    @ccall lib_xgboost.XGCommunicatorIsDistributed()::Cint
end

function XGCommunicatorPrint(message)
    @ccall lib_xgboost.XGCommunicatorPrint(message::Ptr{Cchar})::Cint
end

function XGCommunicatorGetProcessorName(name_str)
    @ccall lib_xgboost.XGCommunicatorGetProcessorName(name_str::Ptr{Ptr{Cchar}})::Cint
end

function XGCommunicatorBroadcast(send_receive_buffer, size, root)
    @ccall lib_xgboost.XGCommunicatorBroadcast(send_receive_buffer::Ptr{Cvoid}, size::Csize_t, root::Cint)::Cint
end

function XGCommunicatorAllreduce(send_receive_buffer, count, data_type, op)
    @ccall lib_xgboost.XGCommunicatorAllreduce(send_receive_buffer::Ptr{Cvoid}, count::Csize_t, data_type::Cint, op::Cint)::Cint
end

# Skipping MacroDefinition: XGB_DLL XGB_EXTERN_C __attribute__ ( ( visibility ( "default" ) ) )

# exports
const PREFIXES = ["XG"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
