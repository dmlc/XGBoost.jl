module XGBoostTermExt

using XGBoost
using XGBoost: Node, OrderedDict, children, getnrounds, isgpu, nfeatures, nrows, ncols
import Term

function _features_display_string(fs, n) 
    str = "{bold yellow}Features:{/bold yellow} "
    if isempty(fs)
        str*"$n (unknown names)"
    else
        string(str, fs)
    end
end

function Term.Panel(dm::DMatrix)
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
        )
end

function Term.Panel(b::Booster)
    info = isempty(b.params) ? () : (paramspanel(b.params),)
    Term.Panel(_features_display_string(b.feature_names, nfeatures(b)),
          info...;
          style="magenta",
          title="XGBoost.Booster",
          title_style="bold cyan",
          subtitle="boosted rounds: $(getnrounds(b))",
          subtitle_style="blue",
         )
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

function Term.Tree(node::Node)
    td = isempty(children(node)) ? Dict(repr(node)=>"leaf") : _tree_display(node)
    Term.Tree(td;
              title="XGBoost Tree (from this node)",
              title_style="bold green",
             )
end

function Term.Panel(node::Node)
    subtitle = if isempty(children(node))
        "{bold green}leaf{/bold green}"
    else
        string(length(children(node)), " children")
    end

    Term.Panel(paramstable(node),
          Term.Tree(node);
          style="magenta",
          title="XGBoost.Node {italic blue}(id=$(node.id), depth=$(node.depth)){/italic blue}",
          title_style="bold cyan",
          subtitle,
          subtitle_style="blue",
         )
end
function paramstable(node::Node)
    if isempty(children(node))
        _paramstable(node, :cover, :leaf)
    else
        _paramstable(node, :split_condition, :yes, :no, :nmissing, :gain, :cover)
    end
end
function _paramstable(node::Node, names::Symbol...)
    vals = mapreduce(Base.Fix1(getproperty, node), hcat, names)
    Term.Table(vals;
               header=map(string, names),
               header_style="bold yellow",
               box=:SIMPLE
              )
end

function XGBoost.importancereport(b::Booster)
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
_importance_number_string(imp) = repr(imp, context=:compact=>true)
_importance_number_string(::Missing) = "{dim}missing{/dim}"

function _tree_display_branch_string(node, child_id::Integer)
    if node.yes == child_id
        string(node.split, " < ", round(node.split_condition, digits=3))
    else
        string(node.split, " ≥ ", round(node.split_condition, digits=3))
    end
end

function _tree_display(node::Node)
    ch = children(node)
    if isempty(ch)
        repr(node; context=:compact=>true)
    else
        OrderedDict(_tree_display_branch_string(node, ch[j].id)=>_tree_display(ch[j]) for j ∈ 1:length(ch))
    end
end

end # module
