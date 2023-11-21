using XGBoost
using CUDA: has_cuda, cu
import Term
using Random, SparseArrays
using Test
using OrderedCollections

include("utils.jl")

@testset "XGBoost" begin

# note that non-Float32 matrices will get truncated and `==` may not hold
@testset "DMatrix Constructors" begin
    X = randn(Float32, 10, 3)
    dm = DMatrix(X)
    @test size(dm) == (10,3)
    @test !XGBoost.hasdata(dm)
    @test dm == X

    X = transpose(randn(Float32, 4, 5))
    dm = DMatrix(X)
    @test (size(dm,1), size(dm,2)) == (5,4)
    @test dm == X

    X = randn(Float32, 4, 5)'
    dm = DMatrix(X)
    @test size(dm) == (5,4)
    @test dm == X

    X = sprand(Float32, 100, 10, 0.1)
    dm = DMatrix(X)
    @test dm == X

    X = [1 2 missing
         3 missing 4
         5 6 7
         missing missing missing]
    dm = DMatrix(X)
    @test isequal(X, dm)

    X = transpose(sprand(Float32, 100, 10, 0.1))
    dm = DMatrix(X)
    @test X == dm

    dm = DMatrix(randn(3,2), Float32[1.0, 2.0, 3.0])
    @test XGBoost.getlabel(dm) == Float32[1.0, 2.0, 3.0]

    dm = DMatrix(randn(3,2), label=Float64[1.0, 2.0, 3.0])
    @test XGBoost.getlabel(dm) ≈ [1.0, 2.0, 3.0]

    dm = DMatrix(randn(3,2), weight=Float32[1.0, 2.0, 3.0])
    @test XGBoost.getweights(dm) ≈ [1.0, 2.0, 3.0]

    # make sure we test retrieving multi-character strings
    tbl = (spock=randn(10), kirk=randn(10), bones=randn(10))
    dm = DMatrix(tbl)
    @test size(dm) == (10,3)
    @test XGBoost.getfeaturenames(dm) == ["spock", "kirk", "bones"]

    itr = [(X=randn(3,2), y=[1,2,3]), (X=randn(3,2), y=[4,5,6])]
    dm = XGBoost.fromiterator(DMatrix, itr)
    @test size(dm) == (6, 2)
    @test XGBoost.getlabel(dm) ≈ 1:6

    tbl = (a=randn(10), b=randn(10))
    y = randn(Float32, 10)
    dm = XGBoost.DMatrix((tbl, y))
    @test size(dm) == (10,2)
    @test XGBoost.getlabel(dm) ≈ y

    tbl = (a=randn(10), b=randn(10), c=randn(Float32, 10))
    dm = XGBoost.DMatrix(tbl, :c)
    @test size(dm) == (10,2)
    @test XGBoost.getlabel(dm) ≈ tbl.c
end

@testset "DMatrix IO" begin
    for (fname, sz) ∈ [("agaricus.txt.train", (6513, 126)), ("agaricus.txt.test", (1611, 126))]
        dm = XGBoost.load(DMatrix, testfilepath(fname), format=:libsvm)
        @test size(dm) == sz

        (X, y) = readlibsvm(testfilepath(fname), sz)
        @test XGBoost.getlabel(dm) == y
    end

    (X, y) = (randn(3,2), 1.0:3.0)
    dm = DMatrix((X, y))
    fname = tempname()
    XGBoost.save(dm, fname)
    dm′ = XGBoost.load(DMatrix, fname, format=:binary)
    @test size(dm) == size(dm′)
    @test XGBoost.getlabel(dm) == XGBoost.getlabel(dm′)
    isfile(fname) && rm(fname)
end

@testset "Agaricus training" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"), format=:libsvm)
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"), format=:libsvm)
    watchlist = Dict("eval"=>dtest, "train"=>dtrain)

    bst = @test_logs (:info, r"XGBoost") (:info, r"") (:info, r"") (:info, r"Training") begin
        xgboost(dtrain, num_round=2,
                watchlist=watchlist,
                η=1, max_depth=2,
                objective="binary:logistic",
                # check that we can set multiple param values
                eval_metric=["rmse", "rmsle"],
               )
    end

    @test XGBoost.nfeatures(bst) == 126

    ŷ = predict(bst, dtest)
    y = XGBoost.getlabel(dtest)
    @test size(y) == size(ŷ)

    δ = count(!iszero, (ŷ .> 0.5) .≠ y) / length(ŷ)
    @test δ < 0.1

    @test length(trees(bst)) == 2

    @testset "custom objective" begin
        ℓ = (ŷ, y) -> (ŷ - y)^2
        ℓ′ = (ŷ, y) -> 2.0*(ŷ - y)
        ℓ″ = (ŷ, y) -> 2.0

        bst = xgboost(dtrain, ℓ′, ℓ″, watchlist=Dict(), num_round=3)

        # we are just checking that the above rand without error and didn't return anything crazy
        @test length(trees(bst)) == 3
    end
end


@testset "Early Stopping rounds" begin

    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"), format=:libsvm)
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"), format=:libsvm)
    # test the early stopping rounds interface with a Dict data type in the watchlist
    watchlist = Dict("eval"=>dtest, "train"=>dtrain)
    
    bst = xgboost(dtrain, 
        num_round=30,
        watchlist=watchlist,
        η=1,
        objective="binary:logistic",
        eval_metric=["rmsle","rmse"]
        )
    
    # test if it ran all the way till the end (baseline)
    nrounds_bst = XGBoost.getnrounds(bst)
    @test nrounds_bst == 30 

    let err = nothing
        try
            # Check to see that xgboost will error out when watchlist supplied is a dictionary with early_stopping_rounds enabled
            bst_early_stopping = xgboost(dtrain,
                num_round=30,
                watchlist=watchlist,
                η=1,
                objective="binary:logistic",
                eval_metric=["rmsle","rmse"],
                early_stopping_rounds = 2
                )

            nrounds_bst = XGBoost.getnrounds(bst) 
            nrounds_bst_early_stopping = XGBoost.getnrounds(bst_early_stopping) 
        catch err
        end

        @test err isa Exception
    end

    # test the early stopping rounds interface with an OrderedDict data type in the watchlist
    watchlist_ordered = OrderedDict("train"=>dtrain, "eval"=>dtest)

    bst_early_stopping = xgboost(dtrain,
        num_round=30,
        watchlist=watchlist_ordered,
        η=1,
        objective="binary:logistic",
        eval_metric=["rmsle","rmse"],
        early_stopping_rounds = 2
        )

    watchlist_nt = (train=dtrain, eval=dtest)
     
    bst_early_stopping = xgboost(dtrain,
        num_round=30,
        watchlist=watchlist_nt,
        η=1,
        objective="binary:logistic",
        eval_metric=["rmsle","rmse"],
        early_stopping_rounds = 2
        )

    @test XGBoost.getnrounds(bst_early_stopping) > 2
    @test XGBoost.getnrounds(bst_early_stopping) <= nrounds_bst

    # get the rmse difference for the dtest
    ŷ = predict(bst_early_stopping, dtest, ntree_limit = bst_early_stopping.best_iteration)

    filename = "agaricus.txt.test"
    lines = readlines(testfilepath(filename))
    y = [parse(Float64,split(s)[1]) for s in lines]

    function calc_rmse(y_true::Vector{T}, y_pred::Vector{T}) where T <: Float64
        return sqrt(sum((y_true .- y_pred).^2)/length(y_true))
    end

    calc_metric = calc_rmse(Float64.(y), Float64.(ŷ))

    # ensure that the results are the same (as numerically possible) with the best round
    @test abs(bst_early_stopping.best_score - calc_metric) < 1e-9

    # test the early stopping rounds interface with an OrderedDict data type in the watchlist using num_parallel_tree parameter
    # this will test the XGBoost API for iteration_range is being utilised properly
    watchlist_ordered = OrderedDict("train"=>dtrain, "eval"=>dtest)

    bst_early_stopping = xgboost(dtrain,
        num_round=30,
        watchlist=watchlist_ordered,
        η=1,
        objective="binary:logistic",
        eval_metric=["rmsle","rmse"],
        early_stopping_rounds = 2,
        num_parallel_tree = 10,
        colsample_bylevel = 0.5
        )

    @test XGBoost.getnrounds(bst_early_stopping) > 2
    @test XGBoost.getnrounds(bst_early_stopping) <= nrounds_bst

    # get the rmse difference for the dtest
    ŷ = predict(bst_early_stopping, dtest, ntree_limit = bst_early_stopping.best_iteration)
    calc_metric = calc_rmse(Float64.(y), Float64.(ŷ))

    # ensure that the results are the same (as numerically possible) with the best round
    @test abs(bst_early_stopping.best_score - calc_metric) < 1e-9

    # Test the interface with no watchlist provided (it'll default to training watchlist)
    let err = nothing
        try
            bst_early_stopping = xgboost(dtrain,
                num_round=30,
                η=1,
                objective="binary:logistic",
                eval_metric=["rmsle","rmse"],
                early_stopping_rounds = 2
                )
        catch err
        end

        @test !(err isa Exception)
    end
end


@testset "Blobs training" begin
    (X, y) = load_classification()

    bst = xgboost((X, y), num_round=10, objective="multi:softprob", num_class=3, watchlist=Dict())

    ŷ = map(ζ -> argmax(ζ) - 1, eachrow(predict(bst, X)))

    # this is a pretty low bar that xgboost should always pass
    @test sum(ŷ .== y)/length(y) > 0.9
end

@testset "Feature importance" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"), format=:libsvm)
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"), format=:libsvm)

    bst = xgboost(dtrain, num_round=5,
                  η=1.0, max_depth=2,
                  objective="binary:logistic",
                  watchlist=Dict(),
                 )

    gain = importance(bst, "gain")
    @test first(keys(gain)) == 29

    weight = importance(bst, "weight")
    @test 29 ∈ keys(weight)

    tbl = importancetable(bst)
    @test tbl.feature[1] == 29
    @test XGBoost.Tables.columnnames(tbl) == (:feature, :gain, :weight, :cover, :total_gain, :total_cover)

    @test typeof(importancereport(bst)) <: Term.Tables.Table
end

# these just ensure we don't have any exceptions
@testset "Term extension" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"), format=:libsvm)
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"), format=:libsvm)

    bst = xgboost(dtrain, num_round=5,
                  η=1.0, max_depth=2,
                  objective="binary:logistic",
                  watchlist=Dict(),
                 )

    @test Term.Panel(dtrain) isa Term.Panel
    @test Term.Panel(bst) isa Term.Panel
end

@testset "Booster" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"), format=:libsvm)
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"), format=:libsvm)

    (model_file, _) = mktemp()

    bst = xgboost(dtrain, num_round=5,
                  η=1.0, max_depth=2,
                  objective="binary:logistic",
                  watchlist=Dict(),
                  eval_metric=("mae", "mape"),
                 )
    preds = predict(bst, dtest)
    XGBoost.save(bst, model_file)

    bst2 = Booster(DMatrix[])
    XGBoost.load!(bst2, model_file)
    @test preds == predict(bst2, dtest)

    bst2 = Booster(DMatrix[])
    XGBoost.load!(bst2, open(model_file))
    @test preds == predict(bst2, dtest)

    bst2 = XGBoost.load(Booster, model_file)
    @test preds == predict(bst2, dtest)

    bst2 = XGBoost.load(Booster, open(model_file))
    @test preds == predict(bst2, dtest)

    buf = XGBoost.save(bst, Vector{UInt8}; format="json")
    bst2 = Booster(DMatrix[])
    XGBoost.load!(bst2, buf)
    @test preds == predict(bst2, dtest)

    bin = XGBoost.serialize(bst)
    bst2 = Booster(DMatrix[])
    XGBoost.deserialize!(bst2, bin)
    @test preds == predict(bst2, dtest)

    # libxgboost re-uses the prediction memory location,
    # so we are testing to make sure we don't do that
    rng = MersenneTwister(999)  # note that Xoshiro is not available on 1.6
    (X, y) = (randn(rng, 10,2), randn(rng, 10))
    b = xgboost((X, y))
    ŷ = predict(b, X)
    @test predict(b, randn(MersenneTwister(998), 10,2)) ≠ ŷ
end

has_cuda() && @testset "cuda" begin
    @info("runing CUDA tests")

    X = randn(Float32, 4, 5)
    dm = DMatrix(cu(X))
    @test size(dm) == size(X)
    @test XGBoost.isgpu(dm)
    @test dm == Matrix(X)

    X = randn(Float32, 4, 5)
    dm = DMatrix(cu(X)')
    @test size(dm) == size(X')
    @test XGBoost.isgpu(dm)
    @test dm == Matrix(X')

    X₀ = randn(Float32, 100, 3)
    X = (x1=cu(X₀[:,1]), x2=cu(X₀[:,2]), x3=cu(X₀[:,3]))
    dm = DMatrix(X)
    @test size(dm) == size(X₀)
    @test XGBoost.isgpu(dm)
    @test dm == X₀
end


end
