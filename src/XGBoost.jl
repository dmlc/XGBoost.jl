__precompile__()

module XGBoost

include("libxgboost.jl")
include("data.jl")
include("learning.jl")

export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model, importance

end # module XGBoost
