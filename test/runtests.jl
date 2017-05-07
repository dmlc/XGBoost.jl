using XGBoost
using Base.Test

include(Pkg.dir("XGBoost") * "/test/utils.jl")

@testset "XGBoost.jl" begin
    @testset "C_API wrapper" begin
        bst = Booster()
        set_attr(bst, test = 1)
        @test attr(bst, "test") == "1"
        set_attr(bst, test = 2)
        @test attributes(bst)["test"] == "2"
        set_attr(bst, test = nothing)
        @test attr(bst, "test") == ""
    end

    @testset "Training interface" begin
        dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
        dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")

        params = Dict("objective" => "binary:logistic",
                      "eta" => 1,
                      "max_depth" => 2,
                      "silent" => 1,
                      "eval_metric" => ["error", "auc"])

        evals = [(dtrain, "train"), (dtest, "test")]
        XGBoost.train(params, dtrain; num_boost_round = 100, evals = evals, verbose_eval = true,
                      early_stopping_rounds = 2)
    end

    @testset "Sparse matrices" begin
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

    @testset "DMatrix loading" begin
        dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
        train_X, train_Y = readlibsvm(Pkg.dir("XGBoost") * "/data/agaricus.txt.train", (6513, 126))
        @test dtrain != nothing

        labels = get_label(dtrain)

        @test train_Y == labels
    end

    @testset "Agaricus training" begin
        dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
        dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")
        watchlist = [(dtest, "eval"), (dtrain, "train")]

        bst = xgboost(dtrain, 2, watchlist=watchlist, eta = 1, max_depth = 2,
                      objective = "binary:logistic", silent = 1)
        @test bst != nothing

        preds = XGBoost.predict(bst, dtest)

        labels = get_label(dtest)
        @test size(preds) == size(labels)

        err = countnz((preds .> 0.5) .!= labels) / length(preds)
        @test err < 0.1
    end

    @testset "Cross validation" begin
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

    @testset "Feature importance" begin
        dtrain = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.train")
        dtest = DMatrix(Pkg.dir("XGBoost") * "/data/agaricus.txt.test")
        watchlist = [(dtest, "eval"), (dtrain, "train")]

        bst = xgboost(dtrain, 5, watchlist = watchlist, eta = 1, max_depth = 2,
                      objective = "binary:logistic", silent = 1, seed = 12345)
        important_features = importance(bst)

        @test startswith(important_features[1].fname, "f28") == true
    end

    @testset "Example is running" begin
        include("example.jl")
    end
end
