using XGBoost

###
# advanced: cutomsized loss function
#

const DATAPATH = joinpath(@__DIR__, "../data")
dtrain = DMatrix(joinpath(DATAPATH, "agaricus.txt.train"))
dtest = DMatrix(joinpath(DATAPATH, "agaricus.txt.test"))

# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing

param = ["max_depth"=>2, "eta"=>1, "silent"=>1]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 2

function logregobj(preds::Vector{Float32}, dtrain::DMatrix)
    labels = get_info(dtrain, "label")
    preds = 1.0 ./ (1.0 + exp(-preds))
    grad = preds - labels
    hess = preds .* (1.0-preds)
    return (grad, hess)
end

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
function evalerror(preds::Vector{Float32}, dtrain::DMatrix)
    labels = get_info(dtrain, "label")
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return ("self-error", sum((preds .> 0.0) .!= labels) / float(size(preds)[1]))
end


# training with customized objective, we can also do step by step training
# simply look at xgboost_lib.jl's implementation of train
bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist,
              obj=logregobj, feval=evalerror)

