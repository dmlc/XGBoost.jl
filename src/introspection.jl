
# available feature type options:
#   weight, gain, cover, total_gain, total_cover

# they seem to only ever name features this way, regardless of whether you set them
_parse_out_feature_name(str::AbstractString) = parse(Int, lstrip(str, 'f'))+1

#TODO: still have to figure out their refusal to name features despite nominally supporting it

function importance(b::Booster, type::AbstractString="gain")
    getnrounds(b) == 0 && return OrderedDict{String,Vector{Float32}}()

    cfg = JSON3.write(Dict("importance_type"=>type))

    olen_fnames = Ref{Lib.bst_ulong}()
    o_fnames = Ref{Ptr{Ptr{Cchar}}}()

    olen_shape = Ref{Lib.bst_ulong}()
    o_shape = Ref{Ptr{Lib.bst_ulong}}()

    o = Ref{Ptr{Cfloat}}()

    xgbcall(XGBoosterFeatureScore, b.handle, cfg, olen_fnames, o_fnames, olen_shape, o_shape, o)

    names = unsafe_wrap(Array, o_fnames[], olen_fnames[])
    names = map(unsafe_string, names)

    dims = unsafe_wrap(Array, o_shape[], olen_shape[])
    dims = reduce(tuple, dims)

    fim = unsafe_wrap(Array, o[], dims)

    if fim isa AbstractVector  # ensure indexing below works regardless of shape
        fim = reshape(fim, (length(fim),1))
    end

    fim = map(j -> fim[j,:], 1:size(fim,1))

    p = sortperm(fim, by=sum, rev=true)

    fns = _parse_out_feature_name.(names[p])
    if length(b.feature_names) ≥ maximum(fns)
        fns = map(j -> b.feature_names[j], fns)
    end

    OrderedDict(fns .=> fim[p])
end


struct Node
    id::Int
    depth::Int

    # the following are only in non-leaf nodes
    split::Union{Nothing,String}
    split_condition::Union{Nothing,Float64}
    yes::Union{Nothing,Int}
    no::Union{Nothing,Int}
    nmissing::Union{Nothing,Int}

    # these are only returned for with_stats
    gain::Union{Nothing,Float64}
    cover::Union{Nothing,Int}  # also in leaves

    # this is the leaf value and only in leaf nodes
    leaf::Union{Nothing,Float64}  # these are leaf values only kept by leaves
    children::Vector{Node}
end

function Node(dict::AbstractDict,
              feature_names::AbstractVector{<:AbstractString}=String[],
              depth::Integer=-1,  # -1 is unknown depth
             )
    depth = if haskey(dict, :depth)
        dict[:depth]
    elseif depth > 0
        depth+1
    else
        error("tried to construct tree node with invalid depth")
    end

    sp = get(dict, :split, nothing)
    if !isnothing(sp) && !isempty(feature_names)
        sp = feature_names[_parse_out_feature_name(sp)]
    end

    ch = get(dict, :children, nothing)
    ch = isnothing(ch) ? Node[] : (ch = map(d -> Node(d, feature_names, depth+1), ch))

    Node(dict[:nodeid],
         depth,
         sp,
         get(dict, :split_condition, nothing),
         get(dict, :yes, nothing),
         get(dict, :no, nothing),
         get(dict, :missing, nothing),
         get(dict, :gain, nothing),
         get(dict, :cover, nothing),
         get(dict, :leaf, nothing),
         ch,
        )
end

trees(b::Booster; with_stats::Bool=true) = map(td -> Node(td, b.feature_names), dump(b; with_stats))

AbstractTrees.NodeType(::Type{Node}) = HasNodeType()
AbstractTrees.nodetype(::Type{Node}) = Node

AbstractTrees.children(n::Node) = n.children
