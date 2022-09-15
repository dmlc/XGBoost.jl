
struct DMatrix
    handle::Ptr{Nothing}

    function DMatrix(handle::Ptr{Nothing})
        dmat = new(handle)
        finalizer(x -> XGDMatrixFree(x.handle), dmat)
        dmat
    end
end

function DMatrix(fname::AbstractString; silent::Integer=false)
    XGDMatrixCreateFromFile(convert(String, fname), convert(Int32, silent)) |> DMatrix
end

function DMatrix(x::Transpose{T}, missing_val::T=convert(T, NaN)) where {T<:Real}
    h = XGDMatrixCreateFromMat(convert(Matrix{Float32}, parent(x)), convert(Float32, missing_val))
    DMatrix(h)
end

function DMatrix(x::AbstractMatrix{T}, missing_val::T=convert(T, NaN)) where {T<:Real}
    DMatrix(transpose(x), missing_val)
end

function DMatrix(x::AbstractMatrix{Union{Missing,T}}) where {T<:Real}
    x′ = map(ξ -> ismissing(ξ) ? NaN32 : Float32(ξ), transpose(x))
    x′ = convert(Matrix{Float32}, x′)
    h = XGDMatrixCreateFromMat(x′, NaN32)
    DMatrix(h)
end

DMatrix(x::SparseMatrixCSC{<:Real,<:Integer}) = XGDMatrixCreateFromCSCEx(x) |> DMatrix

DMatrix(x::Transpose{<:Real,<:SparseMatrixCSC}) = XGDMatrixCreateFromCSCT(parent(x)) |> DMatrix

function setinfo!(dm::DMatrix, name::AbstractString, info::AbstractVector{<:AbstractFloat})
    XGDMatrixSetFloatInfo(dm.handle, convert(String, name),
                          convert(Vector{Float32}, info),
                          convert(UInt64, length(info)),
                         )
    info
end
