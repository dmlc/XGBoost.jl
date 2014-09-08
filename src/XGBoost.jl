module XGBoost

include("xgboost_lib.jl")
export DMatrix, Booster
export xgboost, predict, save, nfold_cv, slice, get_info, set_info, dump_model


end # module
