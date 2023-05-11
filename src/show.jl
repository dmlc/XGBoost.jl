
# note there is a lot more in ext/TermExt.jl

function Base.show(io::IO, dm::DMatrix)
    show(io, typeof(dm))
    print(io, "(", size(dm,1), ", ", size(dm,2), ")")
end

function Base.show(io::IO, b::Booster)
    show(io, typeof(b))
    print(io, "()")
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

"""
    importancereport(b::Booster)

Show a convenient text display of the table output by [`importancetable`](@ref).  This is
intended entirely for display purposes, see [`importance`](@ref) for how to retrieve
feature importance statistics directly.

**NOTE**: Users must have Term.jl loaded before XGBoost.jl for this function to have a method.
"""
function importancereport end
