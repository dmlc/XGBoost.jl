using XGBoost
data = XGDMatrixCreateFromFile("../data/agaricus.txt.train", convert(Int32,0))

return

dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")
param = ["max_depth"=>2, "eta"=>1, "silent"=>0, "objective"=>"binary:logistic"]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 2
bst = xgboost(dtrain, num_round, watchlist=watchlist,
              max_depth=3, eta=1, silent=0, objective = "binary:logistic")
preds = predict(bst, dtest)
save(bst, "0001.model")
save(dtrain, "dtrain.buffer")
save(dtest, "dtest.buffer")
bst2 = Booster("0001.model")
dtest2 = DMatrix("dtest.buffer")
preds2 = predict(bst2, dtest2)
