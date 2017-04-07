using XGBoost


# This script demonstrate how to fit a generalized linear model in XGBoost. This uses a linear
# model instead of trees for the boosters.

dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")

# Change booster to gblinear to fit a linear model
# alpha is the L1 regularizer
# lambda is the L2 regularizer
# you can also set lambda_bias which is L2 regularizer on the bias term

param = ["booster" => "gblinear",
         "eta" => 1,
         "silent" => 0,
         "objective" => "binary:logistic",
         "alpha" => .0001,
         "lambda" => 1]

# Normally, you do not need to set eta (step_size)
# XGBoost uses a parallel coordinate descent algorithm (shotgun),
# there could be affection on convergence with parallelization on certain cases
# setting eta to be smaller value, e.g .5 can make the optimization more stable.

watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 4

bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist)

preds = predict(bst, dtest)
labels = get_label(dtest)

print("test-error=", sum((preds .> 0.5) .!= labels) / float(size(preds)[1]), "\n")
