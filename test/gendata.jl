# this file is for generating test data that's stored in the repository
# it uses MLJ which is not a dependency

# JSON is used as the serialization format only because it is already a dependency

using MLJ
using JSON3


# see utils.jl for loading of the output
function gen_classification()
    (X, y) = make_blobs(1000, 3; centers=3, rng=999)
    y = Int.(int.(y)) .- 1
    X = MLJ.matrix(X)
    fname = joinpath(@__DIR__,"..","assets","data","blobs.json")
    dict = Dict("X1"=>X[:,1], "X2"=>X[:,2], "X3"=>X[:,3], "y"=>y)
    JSON3.write(open(fname, write=true), dict)
end

