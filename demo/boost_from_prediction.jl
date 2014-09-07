using XGBoost

dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")
watchlist  = [(dtest,"eval"), (dtrain,"train")]


###
# advanced: start from a initial base prediction
#

print ("start running example to start from a initial prediction\n")
param = ["max_depth"=>2, "eta"=>1, "silent"=>1, "objective"=>"binary:logistic"]

bst = xgboost(dtrain, 1, param=param, watchlist=watchlist)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation

ptrain = predict(bst, dtrain, output_margin=true)
ptest  = predict(bst, dtest, output_margin=true)

dtrain2 = DMatrix("../data/agaricus.txt.train", base_margin=ptrain)
dtest2 = DMatrix("../data/agaricus.txt.test", base_margin=ptest)
watchlist  = [(dtest2,"eval2"), (dtrain2,"train2")]
print ("this is result of running from initial prediction\n")

bst = xgboost(dtrain2, 1, param=param, watchlist=watchlist)
