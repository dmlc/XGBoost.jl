using XGBoost

# we load in the agaricus dataset
# In this example, we are aiming to predict whether a mushroom can be eated
function svm2dense(fname::ASCIIString, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, float(line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, int(itm[1]) + 1] = float(int(itm[2]))
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end


train = svm2dense("../data/agaricus.txt.train", (6513, 126))
test = svm2dense("../data/agaricus.txt.test", (1611, 126))

#-------------Basic Training using XGBoost-----------------
# this is the basic usage of DMatrix and xgboost
# note: xgboost naturally handles sparse input
# use sparse matrix when your feature is sparse(e.g. when you using one-hot encoding vector)
# model paramters can be set as paramter for ```xgboost``` function, or use an Array{(ASCIIString, Any), 1}/Dict()  
num_round = 2

print("training xgboost with dense matrix\n")
dtrain = DMatrix(train[1], label=train[2])
bst = xgboost(dtrain, num_round, eta=1, max_depth=2)

print("training xgboost with sparse matrix\n")
dtrain = DMatrix(sparse(train[1]), label=train[2])
param = ["max_depth"=>2, "eta"=>1, "silent"=>0, "objective"=>"binary:logistic"]
bst = xgboost(dtrain, num_round, param=param)

print("training xgboost with libsvm format txt\n")
dtrain = DMatrix("../data/agaricus.txt.train")
# To use multi thread, add the param
param["nthread"] = 4
bst = xgboost(dtrain, num_round, param=param)

#--------------------basic prediction using xgboost--------------
# you can do prediction using the following line
# we can set DMatrix without label
dtest = DMatrix(test[1])
labels = test[2]
preds = predict(bst, dtest)
print("test-error=", sum((preds .> 0.5) .!= labels) / float(size(preds)[1]), "\n")

#-------------------save and load models-------------------------
# save model to binary local file
save(bst, "xgb.model")
# load binary model to julia
bst2 = Booster(model_file = "xgb.model")
preds2 = predict(bst2, dtest)
print("sum(abs(pred2-pred))=", sum(abs(preds2 .- preds)), "\n")


#----------------Advanced features --------------
#---------------Using watchlist----------------
# watchlist is a list of DMatrix, each of them tagged with name
# DMatrix in watchlist should have label (for evaluation)
# set label for dtest
set_info(dtest, "label", labels)
watchlist  = [(dtest,"eval"), (dtrain,"train")]
# we can change evaluation metrics, or use multiple evaluation metrics
bst = xgboost(dtrain, num_round, param=param, watchlist=watchlist, metrics=["logloss", "error"])

# we can also save DMatrix into binary file, then we can load it faster next time
save(dtest, "dtest.buffer")
save(dtrain, "dtrain.buffer")

# load model and data in
dtrain = DMatrix("dtrain.buffer")
dtest = DMatrix("dtest.buffer")
bst = Booster(model_file = "xgb.model")

# information can be extracted from DMatrix using get_info
label = get_info(dtest, "label")
pred = predict(bst, dtest)
print("test-error=", sum((pred .> 0.5) .!= label) / float(size(pred)[1]), "\n")


# Finally, you can dump the tree you learned using dump_model into a text file
dump_model(bst, "dump.raw.txt")
# If you have feature map file, you can dump the model in a more beautiul way
dump_model(bst, "dump.nice.txt", fmap="../data/featmap.txt")
