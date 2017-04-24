using XGBoost

dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")

param = ["max_depth" => 2,
         "eta" => 1,
         "silent" => 0,
         "objective" => "binary:logistic"]
watchlist  = [(dtest, "eval"), (dtrain, "train")]
num_round = 3

bst = xgboost(dtrain, num_round, param = param, watchlist = watchlist)


print("start testing prediction from first n trees\n")
labels = get_label(dtest)

### predict using first 1 tree
pred1 = predict(bst, dtest, ntree_limit = 1)
# by default, we predict using all the trees
pred2 = predict(bst, dtest)

print("error of pred1= ", sum((pred1 .> .5) .!= labels) / float(length(pred1)), "\n")
print("error of pred2= ", sum((pred2 .> .5) .!= labels) / float(length(pred2)), "\n")
