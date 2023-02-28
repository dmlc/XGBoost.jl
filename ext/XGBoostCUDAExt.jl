module XGBoostCUDAExt

using XGBoost
using XGBoost: DMatrixHandle, XGDMatrixCreateFromCudaArrayInterface, numpy_json_info, xgbcall
using CUDA: CuMatrix, CuVector

function XGBoost._dmatrix(x::CuMatrix{T}; missing_value::Float32=NaN32, kw...) where {T<:Real}
    o = Ref{DMatrixHandle}()
    cfg = "{\"missing\": $missing_value}"
    GC.@preserve x begin
        info = numpy_json_info(x)
        xgbcall(XGDMatrixCreateFromCudaArrayInterface, info, cfg, o)
    end
    DMatrix(o[]; is_gpu=true, kw...)
end

XGBoost.isa_cuvector(::CuVector) = true

end # module
