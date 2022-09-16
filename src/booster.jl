
mutable struct Booster
    handle::BoosterHandle

    Booster(h::BoosterHandle) = finalizer(x -> xgbcall(XGBoosterFree, x.handle), new(h))
end

function setparam!(b::Booster, name::AbstractString, val::AbstractString)
    xgbcall(XGBoosterSetParam, b.handle, name, val)
    val
end
setparam!(b::Booster, name::AbstractString, val) = setparam!(b, name, string(val))
setparam!(b::Booster, name::Symbol, val) = setparam!(b, string(name), val)

setparams!(b::Booster; kw...) = foreach(kv -> setparam!(b, kv[1], kv[2]), kw)

#TODO: are we sure we are using all threads by default?

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
    setparams!(b; kw...)
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

load(::Type{Booster}, fname::AbstractString) = Booster(DMatrix[], model_file=fanme)

load(::Type{Booster}, io) = Booster(DMatrix[], model_buffer=io)

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

function dump(b::Booster, ::Type{Vector{String}}; fmap::AbstractString="", with_stats::Bool=false)
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Ptr{Cchar}}}()
    xgbcall(XGBoosterDumpModel, b.handle, fmap, convert(Cint, with_stats), olen, o)
    strs = unsafe_wrap(Array, o[], olen[])
    map(unsafe_string, strs)
end

#TODO: may need to serialize model to update? very confused about that, see python code

function serialize(b::Booster)
    olen = Ref{Lib.bst_ulong}()
    o = Ref{Ptr{Int8}}()  # don't know why it insists on Int8
    xgbcall(XGBoosterSerializeToBuffer, b.handle, olen, o)
    unsafe_wrap(Array, convert(Ptr{UInt8}, o[]), olen[])
end

function deserialize(b::Booster, buf::AbstractVector{UInt8})
    buf = convert(Vector{UInt8}, buf)
    xgbcall(XGBoosterUnserializeFromBuffer, b.handle, buf, length(buf))
    b
end

function deserialize(::Type{Booster}, buf::AbstractVector{UInt8}, data...; kw...)
    b = Booster(data...; kw...)
    deserialize(b, buf)
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

function evaliter(b::Booster, watch, n::Integer=1)
    o = Ref{Ptr{Int8}}()
    names = collect(Iterators.map(string, keys(watch)))
    watch = collect(Iterators.map(x -> x.handle, values(watch)))
    xgbcall(XGBoosterEvalOneIter, b.handle, n, watch, names, length(watch), o)
    unsafe_string(o[])
end

printeval(io::IO, b::Booster, watch, n::Integer=1) = print(io, evaliter(b, watch, n))
printeval(b::Booster, watch, n::Integer=1) = printeval(stderr, b, watch, n)

logeval(b::Booster, watch, n::Integer=1) = @info(evaliter(b, watch, n))

function updateone!(b::Booster, Xy::DMatrix;
                    round_number::Integer=1,
                    log_data_name::Union{Nothing,AbstractString}=nothing,
                   )
    xgbcall(XGBoosterUpdateOneIter, b.handle, round_number, Xy.handle)
    isnothing(log_data_name) || logeval(b, Dict(log_data_name=>Xy), round_number)
    b
end

function updateone!(b::Booster, Xy::DMatrix, g::AbstractVector{<:Real}, h::AbstractVector{<:Real};
                    round_number::Integer=1,
                    log_data_name::Union{Nothing,AbstractString}=nothing,
                   )
    if size(g) ≠ size(h)
        throw(ArgumentError("booster got gradient and hessian of incompatible sizes"))
    end
    g = convert(Vector{Cfloat}, g)
    h = convert(Vector{Cfloat}, h)  # uh, why is this not a matrix?
    xgbcall(XGBoosterBoostOneIter(b.handle, Xy.handle, g, h, length(g)))
    isnothing(log_data_name) || logeval(b, Dict(log_data_name=>Xy), round_number)
    b
end

function updateone!(b::Booster, Xy::DMatrix, obj; kw...)
    ŷ = predict(b, Xy)
    (g, h) = obj(ŷ, Xy)
    updateone!(b, Xy, g, h; kw...)
end

function update!(b::Booster, Xy, nrounds::Integer, obj...; kw...)
    for j ∈ 1:nrounds
        updateone!(b, Xy, obj...; round_number=j, kw...)
    end
    b
end
update!(b::Booster, Xy; kw...) = update!(b, Xy, 1; kw...)

function xgboost(data, nrounds::Integer=10;
                 log_data_name::Union{Nothing,AbstractString}="train",
                 kw...)
    Xy = DMatrix(data)
    b = Booster(Xy; kw...)
    isnothing(log_data_name) || @info("XGBoost: starting training:", data)
    update!(b, Xy, nrounds; log_data_name)
    isnothing(log_data_name) || @info("Training rounds complete.")
    b
end


