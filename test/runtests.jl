using XGBoost
using FactCheck

include(Pkg.dir("XGBoost") * "/test/utils.jl")


facts("Sparse matrices") do
    X = sparse(randn(100,10) .* bitrand(100,10))
    y = randn(100)
    DMatrix(X, label = y)

    X = sparse(convert(Matrix{Float32}, randn(10,100) .* bitrand(10,100)))
    y = randn(100)
    DMatrix(X, transposed = true)

    X = sparse(randn(100,10) .* bitrand(100,10))
    y = randn(100)
    DMatrix(X)
end

facts("DMatrix loading") do
    dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
    train_X, train_Y = readlibsvm(Pkg.dir("XGBoost") * "/data/agaricus.txt.train", (6513, 126))
    @fact dtrain --> not(nothing)

    labels = get_label(dtrain)

    @fact train_Y --> labels
end

facts("Agaricus training") do
    dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
    dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = xgboost(dtrain, 2, watchlist=watchlist, eta = 1, max_depth = 2,
                  objective = "binary:logistic", silent = 1)
    @fact bst --> not(nothing)

    preds = XGBoost.predict(bst, dtest)

    labels = get_label(dtest)
    @fact size(preds) --> size(labels)

    err = countnz((preds .> 0.5) .!= labels) / length(preds)
    @fact err --> less_than(0.1)
end

facts("Cross validation") do
    dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
    dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = nfold_cv(dtrain, 5, 3, eta = 1, max_depth = 2, objective = "binary:logistic", silent = 1,
                   seed = 12345)
    # important_features = importance(bst)
    #
    # @fact startswith(important_features[1].fname, "f28") --> true
    # @pending important_features[1].fname --> "f28"
end

facts("Feature importance") do
    dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
    dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = xgboost(dtrain, 5, watchlist = watchlist, eta = 1, max_depth = 2,
                  objective = "binary:logistic", silent = 1, seed = 12345)
    important_features = importance(bst)

    @fact startswith(important_features[1].fname, "f28") --> true
    @pending important_features[1].fname --> "f28"
end

facts("Example is running") do
    include("example.jl")
end
