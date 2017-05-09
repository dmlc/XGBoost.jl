type CVPack
    dtrain::DMatrix
    dtest::DMatrix
    watchlist::Vector{Tuple{DMatrix,String}}
    bst::Booster


    @compat function CVPack(dtrain::DMatrix, dtest::DMatrix, params::Dict{String,<:Any})
        bst = Booster(params = params, cache = [dtrain, dtest])
        watchlist = [(dtrain, "train"), (dtest, "test")]
        return new(dtrain, dtest, watchlist, bst)
    end
end
