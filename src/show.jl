
function Base.show(io::IO, dm::DMatrix)
    show(io, typeof(dm))
    print(io, "(", size(dm,1), ", ", size(dm,2), ")")
end

function Base.show(io::IO, b::Booster)
    show(io, typeof(b))
    print(io, "()")
end

"""
    importancereport(b::Booster)

Show a convenient text display of the table output by [`importancetable`](@ref).

This is intended entirely for display purposes, see [`importance`](@ref) for how to retrieve
feature importance statistics directly.

!!! note
    In Julia >= 1.9, you have to load Term.jl to be able to use this functionality.
"""
function importancereport end

function Base.show(io::IO, node::Node)
    show(io, typeof(node))
    print(io, "(")
    if isempty(children(node))
        print(io, "leaf=", node.leaf)
    else
        print(io, "split_feature=", node.split)
    end
    print(io, ")")
end
