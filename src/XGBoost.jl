__precompile__()

module XGBoost

include("libxgboost.jl")
include("core.jl")
include("training.jl")
include("crossvalidation.jl")
include("legacy.jl")
include("callbacks.jl")

export DMatrix
export get_base_margin, get_float_info, get_label, get_uint_info, get_weight, num_col, num_row,
    save_binary, set_base_margin, set_float_info, set_group, set_label, set_uint_info, set_weight

export Booster
export attr, attributes, boost, dump_model, eval, eval_set, get_dump, load_model, predict,
    save_model, set_attr, set_param

export xgboost, predict, nfold_cv, slice, dump_model, importance

end # module XGBoost
