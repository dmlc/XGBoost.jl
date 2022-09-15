
mutable struct Booster
    handle::BoosterHandle

    Booster(h::BoosterHandle) = finalizer(x -> XGBoosterFree(x.handle), new(h))
end

function setparam!(b::Booster, name::AbstractString, val::AbstractString)
    xgbcall(XGBoosterSetParam, b.handle, name, val)
    val
end
setparam!(b::Booster, name::AbstractString, val) = setparam!(b, name, string(val))
setparam!(b::Booster, name::Symbol, val) = setparam!(b, string(name), val)


function Booster(cache::AbstractVector{<:DMatrix};
                 model_buffer=UInt8[],
                 model_file::AbstractString="",
                 kw...
                )
    o = Ref{BoosterHandle}()
    xgbcall(XGBoosterCreate, map(x -> x.handle, cache), length(cache), o)
    b = Booster(o[])
    if model_buffer isa IO || !isempty(model_buffer)
        load!(b, model_buffer)
    elseif !isempty(model_file)
        load!(b, model_file) 
    end
    foreach(kv -> setparam!(b, kv[1], kv[2]), kw)
    b
end
Booster(dm::DMatrix; kw...) = Booster([dm]; kw...)

load!(b::Booster, file::AbstractString) = (xgbcall(XGBoosterLoadModel, b.handle, file); b)

function load!(b::Booster, buf::AbstractVector{UInt8})
    buf = convert(Vector{UInt8}, buf)
    xgbcall(XGBoosterLoadModelFromBuffer, b.handle, buf, length(buf))
    b
end

load!(b::Booster, io::IO) = load!(b, read(io))

function save(b::Booster, fname::AbstractString)
    xgbcall(XGBoosterSaveModel, b.handle, fname)
    fname
end

function save(b::Booster, ::Type{Vector{UInt8}}; format::AbstractString="json")
    cfg = JSON3.write(Dict("format"=>format))
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{UInt8}}()
    xgbcall(XGBoosterSaveModelToBuffer, cfg, olen, o)
    unsafe_wrap(Array, o[], olen[])
end

save(b::Booster, io::IO; kw...) = write(io, save(b, Vector{UInt8}; kw...))

#TODO: not yet sure if this is working right
function dump(b::Booster, ::Type{Vector{String}}; fmap::AbstractString="", with_stats::Bool=false)
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Ptr{Cchar}}}()
    xgbcall(XGBoosterDumpModel, b.handle, fmap, convert(Cint, with_stats), olen, o)
    strs = unsafe_wrap(Array, o[], olen[])
    map(s -> unsafe_string(s), strs)
end

function predict(b::Booster, Xy::DMatrix;
                 margin::Bool=false,  # whether to output margin
                 training::Bool=false,
                 ntree_limit::Integer=0,  # 0 corresponds to no limit
                )
    opts = margin ? 1 : 0
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Cfloat}}()
    xgbcall(XGBoosterPredict, b.handle, Xy.handle, opts, ntree_limit, Int(training), olen, o)
    unsafe_wrap(Array, o[], olen[])
end
predict(b::Booster, Xy; kw...) = predict(b, DMatrix(Xy); kw...)

#TODO: verbosity when updating

function updateone!(b::Booster, Xy::DMatrix; round_number::Integer=1)
    xgbcall(XGBoostUpdateOneIter, b.handle, round_number, Xy.handle)
    b
end

function updateone!(b::Booster, Xy::DMatrix, g::AbstractVector{<:Real}, h::AbstractVector{<:Real}; 
                    round_number::Integer=1
                   )
    if size(g) ≠ size(h)
        throw(ArgumentError("booster got gradient and hessian of incompatible sizes"))
    end
    g = convert(Vector{Cfloat}, g)
    h = convert(Vector{Cfloat}, h)  # uh, why is this not a matrix?
    xgbcall(XGBoosterBoostOneIter(b.handle, Xy.handle, g, h, length(g)))
    b
end

function updateone!(b::Booster, Xy::DMatrix, obj; kw...)
    ŷ = predict(b, Xy)
    (g, h) = obj(ŷ, Xy)
    updateone!(b, Xy, g, h; kw...)
end

function update!(b::Booster, Xy, nrounds::Integer, obj...; kw...)
    for j ∈ 1:nrounds
        updateone!(b, Xy, obj...; kw...)
    end
    b
end
update!(b::Booster, Xy; kw...) = update!(b, Xy, 1; kw...)

function xgboost(data, nrounds::Integer=10; kw...)
    Xy = DMatrix(data)
    b = Booster(Xy; kw...)
    update!(b, Xy, nrounds)
end
