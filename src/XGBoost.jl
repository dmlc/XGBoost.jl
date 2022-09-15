module XGBoost

using XGBoost_jll

using Printf
using LinearAlgebra
using JSON3
using Random: randperm, MersenneTwister
using SparseArrays: SparseMatrixCSC, nnz
using Statistics: mean, std

export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model, importance

include("Lib.jl")
using .Lib
using .Lib: DMatrixHandle, BoosterHandle

include("dmatrix.jl")
include("booster.jl")

end # module XGBoost
