module XGBoost

using Printf
using Random: randperm, seed!
using SparseArrays: SparseMatrixCSC, nnz
using Statistics: mean, std

export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model, importance
export rabit_init, rabit_finalize, rabit_is_distributed, rabit_get_rank, rabit_get_world_size, rabit_get_version_number

include("../deps/deps.jl")
include("xgboost_wrapper_h.jl")
include("rabit_wrapper.jl")
include("xgboost_lib.jl")

end # module XGBoost
