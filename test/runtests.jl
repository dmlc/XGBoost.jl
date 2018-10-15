using XGBoost
using Random: bitrand
using SparseArrays: sparse
using Test

include("utils.jl")

@testset "XGBoost" begin

@testset "Sparse matrices" begin
    X = sparse(randn(100,10) .* bitrand(100,10))
    y = randn(100)
    DMatrix(X, label = y)

    X = sparse(convert(Matrix{Float32}, randn(10,100) .* bitrand(10,100)))
    y = randn(100)
    DMatrix(X, true)

    X = sparse(randn(100,10) .* bitrand(100,10))
    y = randn(100)
    DMatrix(X)
end

@testset "DMatrix loading" begin
    dtrain = DMatrix("../data/agaricus.txt.train")
    train_X, train_Y = readlibsvm("../data/agaricus.txt.train", (6513, 126))
    @test dtrain != nothing

    labels = get_info(dtrain, "label")

    @test train_Y == labels
end

@testset "Agaricus training" begin
    dtrain = DMatrix("../data/agaricus.txt.train")
    dtest = DMatrix("../data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = xgboost(dtrain, 2, watchlist=watchlist, eta = 1, max_depth = 2,
                  objective = "binary:logistic", silent = 1)
    @test bst != nothing

    preds = XGBoost.predict(bst, dtest)

    labels = get_info(dtest, "label")
    @test size(preds) == size(labels)

    err = count(!iszero, (preds .> 0.5) .!= labels) / length(preds)
    @test err < 0.1
end

@testset "Cross validation" begin
    dtrain = DMatrix("../data/agaricus.txt.train")
    dtest = DMatrix("../data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = nfold_cv(dtrain, 5, 3, eta = 1, max_depth = 2, objective = "binary:logistic", silent = 1,
                   seed = 12345)
    # important_features = importance(bst)
    #
    # @test startswith(important_features[1].fname, "f28")
    # @test important_features[1].fname == "f28" # pending
end

@testset "Feature importance" begin
    dtrain = DMatrix("../data/agaricus.txt.train")
    dtest = DMatrix("../data/agaricus.txt.test")
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    bst = xgboost(dtrain, 5, watchlist = watchlist, eta = 1, max_depth = 2,
                  objective = "binary:logistic", silent = 1, seed = 12345)
    important_features = importance(bst)

    @test startswith(important_features[1].fname, "f28")
    # @test important_features[1].fname --> "f28" # pending
end

@testset "Example" begin
    include("example.jl")
end

end
