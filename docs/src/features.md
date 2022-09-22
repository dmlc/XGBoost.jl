```@meta
CurrentModule = XGBoost
```

# Additional Features


## Introspection

### Feature Importance
This package contains a number of methods for inspecting the results of training and displaying the
results in a legible way with [Term.jl](https://github.com/FedeClaudi/Term.jl).

Feature importances can be computed explicitly using [`importance`](@ref)

For a quick and convenient summary one can use [`importancetable`](@ref).  The output of this
function is primarily intended for visual inspection but it is a Tables.jl compatible table so it
can easily be converted to any tabular format.
```julia
bst = xgboost(X, y, 10)

imp = DataFrame(importancetable(bst))
```

A convenient visualization of this table can also be seen with [`importancereport`](@ref).  These
will use assigned feature names, for example
```julia
julia> df = DataFrame(randn(10,3), ["kirk", "spock", "bones"])
10×3 DataFrame
 Row │ kirk        spock      bones
     │ Float64     Float64    Float64
─────┼───────────────────────────────────
   1 │  0.731406   -0.53631    0.465881
   2 │  0.553427   -0.787531  -0.838059
   3 │  1.30724    -2.38111   -1.1979
   4 │  0.0759902   0.418856   1.49618
   5 │ -0.426773   -0.32008   -0.773329
   6 │ -1.36495    -0.105646   1.08546
   7 │  0.476315   -0.080163  -1.4846
   8 │  0.144403    0.344307  -0.0301839
   9 │  0.593969    0.165502   1.31196
  10 │  2.15151     0.584925  -0.709128

julia> bst = xgboost(df, randn(10), 10)
[ Info: XGBoost: starting training.
[ Info: [1]     train-rmse:0.71749003518059951
[ Info: [2]     train-rmse:0.57348349389049413
[ Info: [3]     train-rmse:0.46118182517533174
[ Info: [4]     train-rmse:0.37161911786076596
[ Info: [5]     train-rmse:0.29986573085749962
[ Info: [6]     train-rmse:0.24238347776088820
[ Info: [7]     train-rmse:0.19544715478958452
[ Info: [8]     train-rmse:0.15795933989281422
[ Info: [9]     train-rmse:0.12805284613811851
[ Info: [10]    train-rmse:0.10467078844629517
[ Info: Training rounds complete.
╭──── XGBoost.Booster ─────────────────────────────────────────────────────────────────╮
│  Features: ["kirk", "spock", "bones"]                                                │
╰──── boosted rounds: 10 ──────────────────────────────────────────────────────────────╯

julia> importancereport(bst)
╭───────────┬────────────┬──────────┬───────────┬──────────────┬───────────────╮
│  feature  │    gain    │  weight  │   cover   │  total_gain  │  total_cover  │
├───────────┼────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "bones"  │  0.229349  │   17.0   │  7.64706  │   3.89893    │     130.0     │
├───────────┼────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "spock"  │  0.176391  │   18.0   │  4.77778  │   3.17503    │     86.0      │
├───────────┼────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "kirk"   │  0.115055  │   13.0   │  3.38462  │   1.49572    │     44.0      │
╰───────────┴────────────┴──────────┴───────────┴──────────────┴───────────────╯
```

### Tree Inspection
The trees of a model belonging to a `Booster` can retrieved and directly inspected with
[`trees`](@ref).

**TODO** FINISH!
