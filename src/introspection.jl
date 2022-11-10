
# available feature type options:
#   weight, gain, cover, total_gain, total_cover

# they seem to only ever name features this way, regardless of whether you set them
_parse_out_feature_name(str::AbstractString) = parse(Int, lstrip(str, 'f'))+1

"""
    importance(b::Booster, type="gain")

Compute feature importance metric from a trained [`Booster`](@ref).
Valid options for `type` are
- `"gain"`
- `"weight"`
- `"cover"`
- `"total_gain"`
- `"total_cover"`

The output is an `OrderedDict` with keys corresponding to feature names and values corresponding
to importances.  The importances are always returned as `Vector`s, typically with length 1 but
possibly longer in multi-class cases.  If feature names were not set the keys of the output dict
will be integers giving the feature column number.  The output will be sorted with the highest
importance feature listed first and the lowest importance feature listed last.

See [`importancetable`](@ref) for a way to generate a tabular summary of all available feature
importances and [`importancereport`](@ref) for a convenient text display of it.
"""
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
    if length(b.feature_names) ≥ maximum(fns, init=0)
        fns = map(j -> b.feature_names[j], fns)
    end

    OrderedDict(fns .=> fim[p])
end


"""
    Node

A data structure representing a node of an XGBoost tree.
These are constructed from the dicts returned by [`dump`](@ref).

`Node`s satisfy the AbstractTrees.jl interface with all nodes being of type
`Node`.

Use `trees(booster)` to return all trees in the model as `Node` objects,
see [`trees`](@ref).

All properties of this struct should be considered public, see
`propertynames(node)` for a list.  Leaf nodes will have their value given
by `leaf`.
"""
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
    cover::Union{Nothing,Float64}  # also in leaves

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
        0  # this case comes up when we have isolated nodes
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

_unwrap_importance(imp) = prod(size(imp)) == 1 ? imp[1] : imp
_unwrap_importance(::Missing) = missing

"""
    importancetable(b::Booster)

Return a Table.jl compatible table (named tuple of `Vector`s) giving a summary of all available
feature importance statistics for `b`.  This table is mainly intended for display purposes,
see [`importance`](@ref) for a more direct way of retrieving importance statistics.
See [`importancereport`](@ref) for a convenient display of this table.
"""
function importancetable(b::Booster)
    imps = (gain=importance(b, "gain"),
            weight=importance(b, "weight"),
            cover=importance(b, "cover"),
            total_gain=importance(b, "total_gain"),
            total_cover=importance(b, "total_cover"),
           )
    fnames = collect(keys(imps.gain))
    𝒻 = dict -> [_unwrap_importance(get(dict, n, missing)) for n ∈ fnames]
    o = (feature=fnames,)
    merge(o, NamedTuple(k=>𝒻(getproperty(imps, k)) for k ∈ propertynames(imps)))
end

"""
    trees(b::Booster; with_stats=true)

Return all trees of the model of the `Booster` `b` as [`Node`](@ref) objects.  The output of this function
is a `Vector` of `Node`s each representing the root of a separate tree.

If `with_stats` the output `Node` objects will contain the computed statistics `gain` and `cover`.
"""
trees(b::Booster; with_stats::Bool=true) = map(td -> Node(td, b.feature_names), dump(b; with_stats))

AbstractTrees.NodeType(::Type{Node}) = HasNodeType()
AbstractTrees.nodetype(::Type{Node}) = Node

AbstractTrees.children(n::Node) = n.children
