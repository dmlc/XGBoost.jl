module XGBoost

using Printf
using Random: randperm, seed!
using SparseArrays: SparseMatrixCSC, nnz
using Statistics: mean, std

export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model, importance

include("xgboost_lib.jl")

end # module XGBoost
