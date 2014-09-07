using XGBoost


##
#  this script demonstrate how to fit generalized linear model in xgboost
#  basically, we are using linear model, instead of tree for our boosters
##

dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")

# change booster to gblinear, so that we are fitting a linear model
# alpha is the L1 regularizer
# lambda is the L2 regularizer
# you can also set lambda_bias which is L2 regularizer on the bias term

param = ["booster"=>"gblinear", "eta"=>1, "silent"=>0,
         "objective"=>"binary:logistic", "alpha"=>0.0001, "lambda"=>1]
##
# the rest of settings are the same
##

watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 4

bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist)

preds = predict(bst, dtest)
labels = get_info(dtest, "label")

print ("error=" , sum((preds .> 0.5)!=labels) /float(size(labels)[1]), "\n")
