
function Base.show(io::IO, dm::DMatrix)
    show(io, typeof(dm))
    print(io, "(", size(dm,1), ", ", size(dm,2), ")")
end

_features_display_string(fs) = "{bold yellow}Features:{/bold yellow} $fs"

function Base.show(io::IO, mime::MIME"text/plain", dm::DMatrix)
    p = Panel(_features_display_string(getfeaturenames(dm)),
              "{dim}(opaque object){/dim}",
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
    Panel(_features_display_string(b.feature_names),
          info...;
          style="magenta",
          title="XGBoost.Booster",
          title_style="bold cyan",
          subtitle="boosted rounds: $(getnrounds(b))",
          subtitle_style="blue",
         )
end

Base.show(io::IO, mime::MIME"text/plain", b::Booster) = show(io, mime, Panel(b))

function _importance_number_string(imp)
    o = prod(size(imp)) == 1 ? imp[1] : imp
    repr(o, context=:compact=>true)
end
_importance_number_string(::Missing) = "{dim}missing{/dim}"

function importancereport(b::Booster)
    if getnrounds(b) == 0
        Panel("{red}(booster not trained){/red}",
              title="XGBoost Feature Importance",
              style="magenta",
             )
    else
        imps = (gain=importance(b, "gain"),
                weight=importance(b, "weight"),
                cover=importance(b, "cover"),
                total_gain=importance(b, "total_gain"),
                total_cover=importance(b, "total_cover"),
               )
        # for now we assume these all have the same feature names
        fnames = collect(keys(imps.gain))
        𝒻 = dict -> [_importance_number_string(get(dict, n, missing)) for n ∈ fnames]
        inner = OrderedDict(:feature=>fnames,
                            :gain=>𝒻(imps.gain),
                            :weight=>𝒻(imps.weight),
                            :cover=>𝒻(imps.cover),
                            :total_gain=>𝒻(imps.total_gain),
                            :total_cover=>𝒻(imps.total_cover),
                           )
        Term.Table(inner,
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
    vals = [getproperty(node, n) for n ∈ names]
    Term.Table(vals';
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

function show(io::IO, node::Node)
    show(io, typeof(node))
    str = if isempty(children(node))
        "leaf=$(node.leaf)"
    else
        "split_feature=$(sprint(show, node.split))"
    end
    print(io, "(", str, ")")
end

show(io::IO, mime::MIME"text/plain", node::Node) = show(io, mime, Panel(node))
