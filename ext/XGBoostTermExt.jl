module XGBoostTermExt

using XGBoost: XGBoost, OrderedDict, children
import Term

function _features_display_string(fs, n) 
    str = "{bold yellow}Features:{/bold yellow} "
    if isempty(fs)
        str*"$n (unknown names)"
    else
        string(str, fs)
    end
end

function Term.Panel(dm::XGBoost.DMatrix; kw...)
    str = if !XGBoost.hasdata(dm)
        "{dim}(values not allocated){/dim}"
    else
        repr(MIME("text/plain"), dm.data; context=:compact=>true)
    end
    subtitle = sprint(dm) do io, dm
        print(io, "(nrows=", XGBoost.nrows(dm), ", ncols=", XGBoost.ncols(dm), ")")
        if XGBoost.isgpu(dm)
            print(io, " {bold green}(GPU){/bold green}")
        end
    end
    Term.Panel(_features_display_string(XGBoost.getfeaturenames(dm), size(dm,2)),
               str;
               style="magenta",
               title="XGBoost.DMatrix",
               title_style="bold cyan",
               subtitle,
               subtitle_style="blue",
               kw...
              )
end

Base.show(io::IO, m::MIME"text/plain", dm::XGBoost.DMatrix) = show(io, m, Term.Panel(dm))

function Term.Panel(b::XGBoost.Booster; kw...)
    info = if isempty(b.params)
        ()
    else
        (paramspanel(b.params; header_style="bold green", columns_style=["bold yellow", "default"], box=:SIMPLE,),)
    end
    Term.Panel(_features_display_string(b.feature_names, XGBoost.nfeatures(b)),
               info...;
               style="magenta",
               title="XGBoost.Booster",
               title_style="bold cyan",
               subtitle="boosted rounds: $(XGBoost.getnrounds(b))",
               subtitle_style="blue",
               kw...
              )
end

function paramspanel(params::AbstractDict; kwargs...)
    names = sort!(collect(keys(params)))
    vals = map(k -> params[k], names)
    Term.Table(OrderedDict(:Parameter=>names, :Value=>vals); kwargs...)
end

Base.show(io::IO, m::MIME"text/plain", b::XGBoost.Booster) = show(io, m, Term.Panel(b))

function Term.Tree(node::XGBoost.Node;
                   title="XGBoost Tree (from this node)",
                   title_style="bold green",
                   kwargs...,
                  )
    td = isempty(children(node)) ? Dict(repr(node)=>"leaf") : _tree_display(node)
    Term.Tree(td; title, title_style, kwargs...)
end

function Term.Panel(node::XGBoost.Node; width::Union{Nothing,Int}=120, kw...)
    subtitle = if isempty(children(node))
        "{bold green}leaf{/bold green}"
    else
        string(length(children(node)), " children")
    end

    Term.Panel(paramstable(node; header_style="bold yellow", box=:SIMPLE),
               Term.Tree(node);
               style="magenta",
               title="XGBoost.Node {italic blue}(id=$(node.id), depth=$(node.depth)){/italic blue}",
               title_style="bold cyan",
               subtitle,
               subtitle_style="blue",
               width,
               kw...
              )
end

Base.show(io::IO, m::MIME"text/plain", node::XGBoost.Node) = show(io, m, Term.Panel(node))

function paramstable(node::XGBoost.Node; kwargs...)
    if isempty(children(node))
        _paramstable(node, :cover, :leaf; kwargs...)
    else
        _paramstable(node, :split_condition, :yes, :no, :nmissing, :gain, :cover; kwargs...)
    end
end
function _paramstable(node::XGBoost.Node, names::Symbol...; kwargs...)
    vals = mapreduce(Base.Fix1(getproperty, node), hcat, names)
    Term.Table(vals; header=map(string, names), kwargs...)
end

function XGBoost.importancereport(b::XGBoost.Booster)
    if XGBoost.getnrounds(b) == 0
        Panel("{red}(booster not trained){/red}",
              title="XGBoost Feature Importance",
              style="magenta",
             )
    else
        tbl = XGBoost.importancetable(b)
        tbl = OrderedDict(k=>_importance_number_string.(getproperty(tbl, k)) for k ∈ propertynames(tbl))
        Term.Table(tbl,
                   header_style="bold green",
                   columns_style=["bold yellow"; fill("default", 5)],
                   box=:ROUNDED,
                  )
    end
end
_importance_number_string(imp) = repr(imp, context=:compact=>true)
_importance_number_string(::Missing) = "{dim}missing{/dim}"

function _tree_display(node::XGBoost.Node)
    ch = children(node)
    if isempty(ch)
        repr(node; context=:compact=>true)
    else
        OrderedDict(_tree_display_branch_string(node, ch[j].id)=>_tree_display(ch[j]) for j ∈ 1:length(ch))
    end
end
function _tree_display_branch_string(node, child_id::Integer)
    if node.yes == child_id
        string(node.split, " < ", round(node.split_condition, digits=3))
    else
        string(node.split, " ≥ ", round(node.split_condition, digits=3))
    end
end

end # module
