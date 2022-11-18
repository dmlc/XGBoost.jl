
function Base.show(io::IO, dm::DMatrix)
    show(io, typeof(dm))
    print(io, "(", size(dm,1), ", ", size(dm,2), ")")
end

function _features_display_string(fs, n) 
    str = "{bold yellow}Features:{/bold yellow} "
    if isempty(fs)
        str*"$n (unknown names)"
    else
        string(str, fs)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", dm::DMatrix)
    str = if !hasdata(dm)
        "{dim}(values not allocated){/dim}"
    else
        sprint((io, x) -> show(io, MIME"text/plain"(), x), dm.data,
               context=:compact=>true,
              )
    end
    p = Panel(_features_display_string(getfeaturenames(dm), size(dm,2)),
              str,
              style="magenta",
              title="XGBoost.DMatrix",
              title_style="bold cyan",
              subtitle="(nrows=$(nrows(dm)), ncols=$(ncols(dm)))",
              subtitle_style="blue",
             )
    show(io, mime, p)
end

function Base.show(io::IO, b::Booster)
    show(io, typeof(b))
    print(io, "()")
end


function paramspanel(params::AbstractDict)
    names = sort!(collect(keys(params)))
    vals = map(k -> params[k], names)
    Term.Table(OrderedDict(:Parameter=>names, :Value=>vals),
               header_style="bold green",
               columns_style=["bold yellow", "default"],
               box=:SIMPLE,
              )
end
paramspanel(b::Booster) = paramspanel(b.params)

function Term.Panel(b::Booster)
    info = isempty(b.params) ? () : (paramspanel(b),)
    Panel(_features_display_string(b.feature_names, nfeatures(b)),
          info...;
          style="magenta",
          title="XGBoost.Booster",
          title_style="bold cyan",
          subtitle="boosted rounds: $(getnrounds(b))",
          subtitle_style="blue",
         )
end

Base.show(io::IO, mime::MIME"text/plain", b::Booster) = show(io, mime, Panel(b))

_importance_number_string(imp) = repr(imp, context=:compact=>true)
_importance_number_string(::Missing) = "{dim}missing{/dim}"

"""
    importancereport(b::Booster)

Show a convenient text display of the table output by [`importancetable`](@ref).  This is
intended entirely for display purposes, see [`importance`](@ref) for how to retrieve
feature importance statistics directly.
"""
function importancereport(b::Booster)
    if getnrounds(b) == 0
        Panel("{red}(booster not trained){/red}",
              title="XGBoost Feature Importance",
              style="magenta",
             )
    else
        tbl = importancetable(b)
        tbl = OrderedDict(k=>_importance_number_string.(getproperty(tbl, k)) for k ∈ propertynames(tbl))
        Term.Table(tbl,
                   header_style="bold green",
                   columns_style=["bold yellow"; fill("default", 5)],
                   box=:ROUNDED,
                  )
    end
end

function _tree_display_branch_string(split, j::Integer)
    o = "($j)"
    isnothing(split) ? o : string(split, " ", o)
end

function _tree_display(node::Node)
    ch = children(node)
    if isempty(ch)
        sprint(show, node)
    else
        OrderedDict(_tree_display_branch_string(ch[j].split, j)=>_tree_display(ch[j]) for j ∈ 1:length(ch))
    end
end

function Term.Tree(node::Node)
    td = isempty(children(node)) ? Dict(sprint(show, node)=>"leaf") : _tree_display(node)
    Term.Tree(td;
              title="XGBoost Tree (from this node)",
              title_style="bold green",
             )
end

function _paramstable(node::Node, names::AbstractVector)
    vals = [getproperty(node, n) for n ∈ permutedims(names)]
    Term.Table(vals;
               header=string.(names),
               header_style="bold yellow",
               box=:SIMPLE
              )
end

_paramstable(node::Node) = _paramstable(node, [:split_condition, :yes, :no, :nmissing, :gain, :cover])

_paramstable_leaf(node::Node) = _paramstable(node, [:cover, :leaf])

paramstable(node::Node) = length(children(node)) == 0 ? _paramstable_leaf(node) : _paramstable(node)

function Term.Panel(node::Node)
    subtitle = if isempty(children(node))
        "{bold green}leaf{/bold green}"
    else
        string(length(children(node)), " children")
    end

    Panel(paramstable(node),
          Term.Tree(node);
          style="magenta",
          title="XGBoost.Node {italic blue}(id=$(node.id), depth=$(node.depth)){/italic blue}",
          title_style="bold cyan",
          subtitle,
          subtitle_style="blue",
         )
end

function Base.show(io::IO, node::Node)
    show(io, typeof(node))
    str = if isempty(children(node))
        "leaf=$(node.leaf)"
    else
        "split_feature=$(sprint(show, node.split))"
    end
    print(io, "(", str, ")")
end

Base.show(io::IO, mime::MIME"text/plain", node::Node) = show(io, mime, Panel(node))
