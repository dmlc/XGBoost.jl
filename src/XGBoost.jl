module XGBoost

using LinearAlgebra
using SparseArrays: SparseMatrixCSC, nnz
using AbstractTrees
using OrderedCollections
using JSON3
using Tables
using Term
using Statistics: mean, std

using Base.Iterators: Stateful, reset!

export DMatrix, Booster
export updateone!, update!, predict, xgboost
export importance, importancetable, importancereport, trees

include("Lib.jl")
using .Lib
using .Lib: DMatrixHandle, BoosterHandle


include("dmatrix.jl")
include("booster.jl")
include("introspection.jl")
include("show.jl")
include("defaultparams.jl")


end # module XGBoost
