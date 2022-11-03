using XGBoost
using Random, SparseArrays
using Test

include("utils.jl")

@testset "XGBoost" begin

# it's *very* hard to do real tests here because we can't get data back
# this section just checks for errors, only training below can verify it's working properly
@testset "DMatrix Constructors" begin
    dm = DMatrix(randn(10,3))
    @test size(dm) == (10,3)

    dm = DMatrix(transpose(randn(4,5)))
    @test (size(dm,1), size(dm,2)) == (5,4)

    dm = DMatrix(randn(4,5)')
    @test size(dm) == (5,4)

    dm = DMatrix(sprand(100, 10, 0.1))
    @test size(dm) == (100,10)

    dm = DMatrix([1 2 missing
                  3 missing 4
                  5 6 7
                  missing missing missing])
    @test size(dm) == (4,3)

    dm = DMatrix(transpose(sprand(100, 10, 0.1)))
    @test size(dm) == (10,100)

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
end

@testset "DMatrix IO" begin
    for (fname, sz) ∈ [("agaricus.txt.train", (6513, 126)), ("agaricus.txt.test", (1611, 126))]
        dm = XGBoost.load(DMatrix, testfilepath(fname))
        @test size(dm) == sz

        (X, y) = readlibsvm(testfilepath(fname), sz)
        @test XGBoost.getlabel(dm) == y
    end

    (X, y) = (randn(3,2), 1.0:3.0)
    dm = DMatrix((X, y))
    fname = tempname()
    XGBoost.save(dm, fname)
    dm′ = XGBoost.load(DMatrix, fname)
    @test size(dm) == size(dm′)
    @test XGBoost.getlabel(dm) == XGBoost.getlabel(dm′)
    isfile(fname) && rm(fname)
end

@testset "Agaricus training" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"))
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"))
    watchlist = Dict("eval"=>dtest, "train"=>dtrain)

    bst = @test_logs (:info, r"XGBoost") (:info, r"") (:info, r"") (:info, r"Training") begin
        xgboost(dtrain, num_round=2,
                watchlist=watchlist,
                η=1, max_depth=2,
                objective="binary:logistic",
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

@testset "Feature importance" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"))
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"))

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

    @test typeof(importancereport(bst)) <: XGBoost.Term.Tables.Table
end

@testset "Booster Save/Load/Serialize" begin
    dtrain = XGBoost.load(DMatrix, testfilepath("agaricus.txt.train"))
    dtest = XGBoost.load(DMatrix, testfilepath("agaricus.txt.test"))

    model_file = "test.xgb"

    bst = xgboost(dtrain, num_round=5,
                  η=1.0, max_depth=2,
                  objective="binary:logistic",
                  watchlist=Dict(),
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
end


end
