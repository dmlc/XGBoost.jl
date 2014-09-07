using XGBoost

dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")

param = ["max_depth"=>2, "eta"=>1, "silent"=>0, "objective"=>"binary:logistic"]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 3

bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist)


print ("start testing prediction from first n trees\n")
label = get_info(dtest, "label")

### predict using first 1 tree
ypred1 = predict(bst, dtest, ntree_limit=1)
# by default, we predict using all the trees
ypred2 = predict(bst, dtest)

print ("error of ypred1=" , sum((ypred1 .> 0.5)!=label) /float(size(label)[1]), "\n")
print ("error of ypred2=" , sum((ypred2 .> 0.5)!=label) /float(size(label)[1]), "\n")

# Not Pass!
