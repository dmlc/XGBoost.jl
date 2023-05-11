module CUDAExt

using XGBoost, LinearAlgebra, Tables, CUDA

using XGBoost: DMatrixHandle, xgbcall
using XGBoost: Node, numpy_json_info, numpy_json_infos
using XGBoost: XGDMatrixCreateFromCudaArrayInterface, XGDMatrixCreateFromCudaColumnar


XGBoost._check_gpu_col(::CuArray) = true

function XGBoost._dmatrix_gpu_table(cols::Tables.Columns; missing_value::Float32=NaN32, kw...)
    o = Ref{DMatrixHandle}()
    cfg = "{\"missing\": $missing_value}"
    GC.@preserve cols begin
        infos = numpy_json_infos(cols)
        xgbcall(XGDMatrixCreateFromCudaColumnar, infos, cfg, o)
    end
    DMatrix(o[]; is_gpu=true, kw...)
end

# sadly we have to copy CuArray because of incompatible column convention
function _transposed_cuda_dmatrix(x::CuArray{T}; missing_value::Float32=NaN32, kw...) where {T<:Real}
    o = Ref{DMatrixHandle}()
    cfg = "{\"missing\": $missing_value}"
    GC.@preserve x begin
        info = numpy_json_info(x)
        xgbcall(XGDMatrixCreateFromCudaArrayInterface, info, cfg, o)
    end
    DMatrix(o[]; is_gpu=true, kw...)
end

XGBoost.DMatrix(x::Transpose{T,<:CuArray}; kw...) where {T<:Real} = _transposed_cuda_dmatrix(parent(x); kw...)
XGBoost.DMatrix(x::Adjoint{T,<:CuArray}; kw...) where {T<:Real} = _transposed_cuda_dmatrix(parent(x); kw...)

function XGBoost.DMatrix(x::CuArray; kw...)
    x′ = CuArray(transpose(x))
    _transposed_cuda_dmatrix(x′; kw...)
end


end
