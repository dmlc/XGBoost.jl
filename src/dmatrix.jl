
#FUCK: matrices are getting scrambled somehow!!!

mutable struct DMatrix
    handle::DMatrixHandle

    function DMatrix(handle::Ptr{Nothing}; kw...)
        dm = new(handle)
        setinfos!(dm; kw...)
        finalizer(x -> xgbcall(XGDMatrixFree, x.handle), dm)
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

setinfo!(dm::DMatrix, name::Symbol, info) = setinfo!(dm, string(name), info)

setinfos!(dm::DMatrix; kw...) = foreach(kv -> setinfo!(dm, kv[1], kv[2]), kw)

_getinfo_call(::Type{<:AbstractFloat}) = XGDMatrixGetFloatInfo
_getinfo_call(::Type{<:Integer}) = XGDMatrixGetUIntInfo

function getinfo(dm::DMatrix, ::Type{T}, name::AbstractString) where {T<:Real}
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Cfloat}}()
    xgbcall(_getinfo_call(T), dm.handle, name, olen, o)
    x = unsafe_wrap(Array, o[], olen[])
    convert(Vector{T}, x)
end

function load(::Type{DMatrix}, fname::AbstractString; silent::Bool=true, kw...)
    o = Ref{DMatrixHandle}()
    xgbcall(XGDMatrixCreateFromFile, fname, silent, o)
    DMatrix(o[], kw...)
end

function _dmatrix(x::AbstractMatrix{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    o = Ref{DMatrixHandle}()
    sz = reverse(size(x))
    xp = convert(Matrix{Cfloat}, x)
    xgbcall(XGDMatrixCreateFromMat, xp, sz[1], sz[2], missing_val, o)
    DMatrix(o[]; kw...)
end

function DMatrix(x::AbstractMatrix{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    # sadly, this copying is unavoidable
    _dmatrix(Matrix(transpose(x)), missing_val; kw...)
end

# ideally these would be recursive but can't be bothered
function DMatrix(x::Transpose{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    _dmatrix(parent(x), missing_val; kw...)
end
function DMatrix(x::Adjoint{T}, missing_val::T=convert(T, NaN); kw...) where {T<:Real}
    _dmatrix(parent(x), missing_val; kw...)
end

function DMatrix(x::AbstractMatrix{Union{Missing,T}}; kw...) where {T<:Real}
    # we try to make it so that we only have to copy once
    x′ = map(ξ -> ismissing(ξ) ? NaN32 : Float32(ξ), transpose(x))
    x′ = convert(Matrix{Cfloat}, x′)
    o = Ref{DMatrixHandle}()
    xgbcall(XGDMatrixCreateFromMat, x′, size(x′,1), size(x′,2), NaN32, o)
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

#TODO: transpose sparse method

# this method takes a slice
function DMatrix(dm::DMatrix, idx::AbstractVector{<:Integer}; kw...)
    o = Ref{DMatrixHandle}()
    idx = convert(Vector{Cint}, idx .- 1)
    XGDMatrixSliceDMatrix(dm.handle, idx, length(idx), o)
    DMatrix(o[]; kw...)
end

# we require the colon for consistent array semantics
function Base.getindex(dm::DMatrix, idx::AbstractVector{<:Integer}, ::Colon; kw...)
    DMatrix(dm, idx; kw...)
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


#====================================================================================================
       proxy stuff

Proxy is for setting data later, typically for external memory stuff
====================================================================================================#

function proxy(::Type{DMatrix})
    o = Ref{DMatrixHandle}()
    xgbcall(XGProxyDMatrixCreate, o)
    DMatrix(o[])
end

_numpy_json_typestr(::Type{<:AbstractFloat}) = "f"
_numpy_json_typestr(::Type{<:Integer}) = "i"
_numpy_json_typestr(::Type{Bool}) = "b"
_numpy_json_typestr(::Type{<:Complex{<:AbstractFloat}}) = "c"

numpy_json_typestr(::Type{T}) where {T<:Number} = string("<",_numpy_json_typestr(T),sizeof(T))

function numpy_json_info(x::Transpose; read_only::Bool=false)
    info = Dict("data"=>(convert(Csize_t, pointer(parent(x))), read_only),
                "shape"=>reverse(size(x)),
                "typestr"=>numpy_json_typestr(eltype(x)),
                "version"=>3,
               )
    JSON3.write(info)
end
numpy_json_info(x::Array) = x |> transpose |> numpy_json_info

#TODO: this is not nearly done... I'm very worried about data ownership

#NOTE: I'm assuming the dmatrix will take ownership but I have no idea if that's true
# maybe our data iterator thing is supposed to take ownership?
function setproxy!(dm::DMatrix, x::AbstractMatrix; kw...)
    x = convert(Matrix, x)
    GC.@preserve x begin
        info = numpy_json_info(x)
        xgbcall(XGProxyDMatrixSetDataDense, dm.handle, info)
    end
    for (k,v) ∈ kw
        setinfo!(dm, string(k), v)
    end
    dm
end
setproxy!(dm::DMatrix, X::AbstractMatrix, y::AbstractVector; kw...) = setproxy!(dm, X; label=y, kw...)


mutable struct DataIterator{T<:Stateful}
    iter::T
    proxy::DMatrix

    DataIterator(iter::Stateful) = new{typeof(iter)}(iter, proxy(DMatrix))
end

DataIterator(x) = DataIterator(Stateful(x))

Iterators.reset!(itr::DataIterator) = reset!(itr.iter)

Base.isempty(itr::DataIterator) = isempty(itr.iter)

function Base.popfirst!(itr::DataIterator)
    x = popfirst!(itr.iter)
    # TODO: make this a little more intelligent
    args = haskey(x, :y) ? (x.X, x.y) : (x.X,)
    kw = (;(k=>v for (k,v) ∈ pairs(x) if k ∉ (:X, :y))...)
    setproxy!(itr.proxy, args...; kw...)
    x
end

function _unsafe_dataiter_next(ptr::Ptr)
    itr = unsafe_pointer_to_objref(ptr)::DataIterator
    try
        if isempty(itr)
            Cint(0)
        else
            popfirst!(itr)
            Cint(1)
        end
    catch err
        @error("got error during C callback for next iteration state", exception=(err, catch_backtrace()))
        Cint(-1)
    end
end

function _unsafe_dataiter_reset(ptr::Ptr)
    itr = unsafe_pointer_to_objref(ptr)::DataIterator
    try
        reset!(itr)
    catch err
        @error("got error during C callback for resetting iterator", exception=(err, catch_backtrace()))
    end
    nothing
end

function _dmatrix_caching_config_json(;cache_prefix::AbstractString,
                                      nthreads::Union{Integer,Nothing},
                                     )
    d = Dict("missing"=>"__NAN_STR__",
             "cache_prefix"=>cache_prefix,
            )
    isnothing(nthreads) || (d["nthreads"] = nthreads)
    # xgboost allows this which JSON3 thinks is invalid
    replace(JSON3.write(d), "\"__NAN_STR__\""=>"NaN")
end

function DMatrix(itr::DataIterator;
                 cache_prefix::AbstractString=joinpath(tempdir(),"xgb-cache"),
                 nthreads::Union{Integer,Nothing}=nothing,
                 kw...
                )
    o = Ref{DMatrixHandle}()

    ptr_rst = @cfunction(_unsafe_dataiter_reset, Cvoid, (Ptr{Cvoid},))
    ptr_next = @cfunction(_unsafe_dataiter_next, Cint, (Ptr{Cvoid},))

    xgbcall(XGDMatrixCreateFromCallback, pointer_from_objref(itr),
            itr.proxy.handle,
            ptr_rst, ptr_next,
            _dmatrix_caching_config_json(;cache_prefix, nthreads),
            o,
           )

    DMatrix(o[]; kw...)
end

# this is so we don't have endless confusing constructor methods
fromiterator(::Type{DMatrix}, itr; kw...) = DMatrix(DataIterator(itr); kw...)
