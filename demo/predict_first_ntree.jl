using XGBoost

const DATAPATH = joinpath(@__DIR__, "../data")
dtrain = DMatrix(joinpath(DATAPATH, "agaricus.txt.train"))
dtest = DMatrix(joinpath(DATAPATH, "agaricus.txt.test"))

param = ["max_depth" => 2,
         "eta" => 1,
         "silent" => 0,
         "objective" => "binary:logistic"]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 3

bst = xgboost(dtrain, num_round, param = param, watchlist = watchlist)


print("start testing prediction from first n trees\n")
labels = get_info(dtest, "label")

### predict using first 1 tree
pred1 = predict(bst, dtest, ntree_limit = 1)
# by default, we predict using all the trees
pred2 = predict(bst, dtest)

print("error of pred1= ", sum((pred1 .> 0.5) .!= labels) / float(size(pred1)[1]), "\n")
print("error of pred2= ", sum((pred2 .> 0.5) .!= labels) / float(size(pred2)[1]), "\n")
