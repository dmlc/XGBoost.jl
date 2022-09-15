
mutable struct DMatrix
    handle::DMatrixHandle

    function DMatrix(handle::Ptr{Nothing}; kw...)
        dmat = new(handle)
        for (k,v) ∈ kw
            setinfo!(dmat, string(k), v)
        end
        finalizer(x -> XGDMatrixFree(x.handle), dmat)
    end
end

function setinfo!(dm::DMatrix, name::AbstractString, info::AbstractVector{<:AbstractFloat})
    xgbcall(XGDMatrixSetFloatInfo, dm.handle, name, convert(Vector{Cfloat}, info), length(info))
    info
end

function setinfo!(dm::DMatrix, name::AbstractString, info::AbstractVector{<:Integer})
    xgbcall(XGDMatrixSetUIntInfo, dm.handle, name, convert(Vector{Cuint}, info), length(info))
    info
end

_getinfo_call(::Type{<:AbstractFloat}) = XGDMatrixGetFloatInfo
_getinfo_call(::Type{<:Integer}) = XGDMatrixGetUIntInfo

function getinfo(dm::DMatrix, ::Type{T}, name::AbstractString) where {T<:Real}
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Cfloat}}()
    xgbcall(_getinfo_call(T), dm.handle, name, olen, o)
    x = unsafe_wrap(Array, o[], olen[])
    convert(Vector{T}, x)
end

function DMatrix(fname::AbstractString; silent::Bool=false, kw...)
    o = Ref{DMatrixHandle}()
    xgbcall(XGDMatrixCreateFromFile, fname, silent, o)
    DMatrix(o[], kw...)
end

function DMatrix(x::Transpose{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    o = Ref{DMatrixHandle}()
    xp = convert(Matrix{Cfloat}, parent(x))
    xgbcall(XGDMatrixCreateFromMat, xp, size(x,2), size(x,1), missing_val, o)
    DMatrix(o[]; kw...)
end

function DMatrix(x::AbstractMatrix{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    DMatrix(transpose(x), missing_val; kw...)
end

function DMatrix(x::AbstractMatrix{Union{Missing,T}}; kw...) where {T<:Real}
    # we try to make it so that we only have to copy once
    x′ = map(ξ -> ismissing(ξ) ? NaN32 : Float32(ξ), transpose(x))
    x′ = convert(Matrix{Cfloat}, x′)
    o = Ref{DMatrixHandle}()
    XGDMatrixCreateFromMat(x′, size(x′,1), size(x′,2), NaN32, o)
    DMatrix(o[]; kw...)
end

function _sparse_csc_components(x::SparseMatrixCSC)
    colptr = convert(Vector{Csize_t}, x.colptr .- 1)
    rowval = convert(Vector{Cuint}, x.rowval .- 1)
    nzval = convert(Vector{Cfloat}, x.nzval)
    (colptr, rowval, nzval)
end

function DMatrix(x::SparseMatrixCSC{<:Real,<:Integer}; kw...)
    o = Ref{DMatrixHandle}()
    (colptr, rowval, nzval) = _sparse_csc_components(x)
    xgbcall(XGDMatrixCreateFromCSCEx, colptr, rowval, nzval,
            size(colptr,1), nnz(x), size(x,1),
            o,
           )
    DMatrix(o[]; kw...)
end

function DMatrix(x::Transpose{<:Real,<:SparseMatrixCSC}; kw...)
    x′ = parent(x)
    o = Ref{DMatrixHandle}()
    (colptr, rowval, nzval) = _sparse_csc_components(x)
    xgbcall(XGDMatrixCreateFromCSREx, colptr, rowval, nzval,
            size(colptr,1), nnz(x), size(x,2),
            o,
           )
    DMatrix(o[]; kw...)
end

# this method takes a slice
function DMatrix(dm::DMatrix, idx::AbstractVector{<:Integer}; kw...)
    o = Ref{DMatrixHandle}()
    idx = convert(Vector{Cint}, idx .- 1)
    XGDMatrixSliceDMatrix(dm.handle, idx, length(idx), o)
    DMatrix(o[]; kw...)
end

DMatrix(X::AbstractMatrix, y::AbstractVector; kw...) = DMatrix(X; label=y, kw...)

DMatrix(Xy::Tuple{TX,Ty}; kw...) where {TX,Ty} = DMatrix(Xy[1], Xy[2]; kw...)

DMatrix(dm::DMatrix) = dm

function nrows(dm::DMatrix)
    o = Ref{Lib.bst_ulong}()
    xgbcall(XGDMatrixNumRow, dm.handle, o)
    convert(Int, o[])
end

function ncols(dm::DMatrix)
    o = Ref{Lib.bst_ulong}()
    xgbcall(XGDMatrixNumCol, dm.handle, o)
    convert(Int, o[])
end

Base.size(dm::DMatrix) = (nrows(dm), ncols(dm))
function Base.size(dm::DMatrix, ax::Integer)
    if ax == 1
        nrows(dm)
    elseif ax == 2
        ncols(dm)
    else
        throw(ArgumentError("size: DMatrix only has 2 indices"))
    end
end

#TODO: show feature names if available (probably only for MIME)
function Base.show(io::IO, dm::DMatrix)
    show(io, typeof(dm))
    print(io, "(", size(dm,1), ", ", size(dm,2), ")")
end

getlabel(dm::DMatrix) = getinfo(dm, Float32, "label")

function save(dm::DMatrix, fname::AbstractString; silent::Bool=true)
    xgbcall(XGDMatrixSaveBinary, dm.handle, fname, convert(Cint, silent))
    fname
end

function setfeatureinfo!(dm::DMatrix, k::AbstractString, strs::AbstractVector{<:AbstractString})
    strs = convert(Vector{String}, strs)
    xgbcall(XGDMatrixSetStrFeatureInfo, dm.handle, k, strs, length(strs))
    strs
end

setfeaturenames!(dm::DMatrix, names::AbstractVector{<:AbstractString}) = setfeatureinfo!(dm, "feature_name", names)

function getfeatureinfo(dm::DMatrix, k::AbstractString)
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Ptr{Cchar}}}()
    xgbcall(XGDMatrixGetStrFeatureInfo, dm.handle, k, olen, o)
    strs = unsafe_wrap(Array, o[], olen[])
    map(unsafe_string, strs)
end

getfeaturenames(dm::DMatrix) = getfeatureinfo(dm, "feature_name")


#TODO: Tables.jl methods (input only)
