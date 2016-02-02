using XGBoost
using FactCheck

include("utils.jl")

facts("DMatrix loading") do
    dtrain = DMatrix("../data/agaricus.txt.train")
    train_X, train_Y = readlibsvm("../data/agaricus.txt.train", (6513, 126))
    @fact dtrain --> not(nothing)

    labels = get_info(dtrain, "label")

    @fact train_Y --> labels
end


facts("Agaricus training") do
    dtrain = DMatrix("../data/agaricus.txt.train")
    dtest = DMatrix("../data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = xgboost(dtrain, 2, watchlist=watchlist,  eta=1, max_depth=2, objective="binary:logistic")
    @fact bst --> not(nothing)

    preds = XGBoost.predict(bst, dtest)

    labels = get_info(dtest, "label")
    @fact size(preds) --> size(labels)

    err = countnz((preds .> 0.5) .!= labels) / length(preds)
    @fact err --> less_than(0.1)
end


facts("Example is running") do
    include("example.jl")
end
