module XGBoost

using XGBoost_jll

using Printf
using LinearAlgebra
using Random: randperm, MersenneTwister
using SparseArrays: SparseMatrixCSC, nnz
using Statistics: mean, std

export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model, importance
export rabit_init, rabit_finalize, rabit_is_distributed, rabit_get_rank, rabit_get_world_size, rabit_get_version_number

include("wrapper.jl")
include("rabit_wrapper.jl")
include("dmatrix.jl")

end # module XGBoost
