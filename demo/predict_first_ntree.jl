using XGBoost

dtrain = DMatrix("../data/agaricus.txt.train")
dtest = DMatrix("../data/agaricus.txt.test")

param = ["max_depth"=>2, "eta"=>1, "silent"=>0, "objective"=>"binary:logistic"]
watchlist  = [(dtest,"eval"), (dtrain,"train")]
num_round = 3

bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist)


print ("start testing prediction from first n trees\n")
labels = get_info(dtest, "label")

### predict using first 1 tree
pred1 = predict(bst, dtest, ntree_limit=1)
# by default, we predict using all the trees
pred2 = predict(bst, dtest)

tmp = zip(pred1, labels)
cnt = 0
for itm in tmp
    if convert(Integer, itm[1] > 0.5) != itm[2]
        cnt += 1
    end
end
print("error of pred1= ", string(cnt / convert(Real, size(labels)[1])), "\n")

tmp = zip(pred2, labels)
cnt = 0
for itm in tmp
    if convert(Integer, itm[1] > 0.5) != itm[2]
        cnt += 1
    end
end

print("error of pred2= ", string(cnt / convert(Real, size(labels)[1])), "\n")
