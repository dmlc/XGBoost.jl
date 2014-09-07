module XGBoost

include("xgboost_lib.jl")
export xgboost, predict, save, nfold_cv, slice, Booster, DMatrix, get_info, dump_model


end # module
