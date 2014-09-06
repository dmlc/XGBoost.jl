using XGBoost
dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")
param = ["max_depth"=>2, "eta"=>1, "silent"=>0, "objective"=>"binary:logistic"]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 2
bst = xgboost( param, dtrain, num_round, evals=watchlist)
preds = predict(bst, dtest)
save(bst, "0001.model")
save(dtrain, "dtrain.buffer")
save(dtest, "dtest.buffer")
bst2 = Booster("0001.model")
dtest2 = DMatrix("dtest.buffer")
preds2 = predict(bst2, dtest2)
