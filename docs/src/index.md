```@meta
CurrentModule = XGBoost
```

# XGBoost

This is the Julia wrapper of the [xgboost](https://xgboost.ai/) gradient boosting
library.

## TL;DR
```julia
using XGBoost

# training set of 100 datapoints of 4 features
(X, y) = (randn(100,4), randn(100))

# create and train a gradient boosted tree model of 5 trees
bst = xgboost((X, y), num_round=5, max_depth=6, objective="reg:squarederror")

# obtain model predictions
ŷ = predict(bst, X)


using DataFrames
df = DataFrame(randn(100,3), [:a, :b, :y])

# can accept tabular data, will keep feature names
bst = xgboost((df[!, [:a, :b]], df.y))

# display importance statistics retaining feature names
importancereport(bst)

# return AbstractTrees.jl compatible tree objects describing the model
trees(bst)
```

## Data Input
Data is passed to xgboost via the [`DMatrix`](@ref) object.  This is an
`AbstractMatrix{Union{Missing,Float32}}` object which is primarily intended for internal use by
`libxgboost`.  Julia `AbstractArray` data will automatically be wrapped in a `DMatrix` where
appropriate, so users should mostly not have to call its constructors directly, but it may be
helpful to understand the semantics for creating it.

For example, the following are equivalent
```julia
X = randn(4,3)
predict(bst, X) == predict(bst, DMatrix(X))
```

The xgboost library interprets floating point `NaN` values as "missing" or "null" data.  `missing`
values will automatically be converted so that the semantics of the resulting `DMatrix` will match
that of a provided Julia matrix with `Union{Missing,Real}` values.  For example
```julia
X = [0 missing 1
     1 0 missing
     missing 1 0]
isequal(DMatrix(X), x)  # nullity is preserved
```

!!! note

    `DMatrix` must allocate new arrays when fetching values from it.  One therefore should avoid
    using `DMatrix` directly except with `XGBoost`; retrieving values from this object should be
    considered useful mostly only for verification.


### Feature Naming and Tabular Data
Xgboost supports the naming of features (i.e. columns of the feature matrix).  This can be useful
for inspecting trained models.
```julia
X = randn(10,3)

dm = DMatrix(X, feature_names=["a", "b", "c"])

XGBoost.setfeaturenames!(dm, ["a", "b", "c"])  # can also set after construction
```

`DMatrix` can also accept tabular arguments.  These can be any table that satisfies the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface (e.g. a `NamedTuple` of same-length
`AbstractVector`s or a `DataFrame`).
```julia
using DataFrames
df = DataFrame(randn(10,3), [:a, :b, :c])

y = randn(10)

DMatrix(df, y)

df[!, :y] = y
DMatrix(df, :y)  # equivalent to DMatrix(df, y)
```

When constructing a `DMatrix` from a table the feature names will automatically be set to the names
of the columns (this can be overridden with the `feature_names` keyword argument).
```julia
df = DataFrame(randn(10,3), [:a, :b, :c])
dm = DMatrix(df)
XGBoost.getfeaturenames(dm) == ["a", "b", "c"]
```

### Label Data
Since xgboost is a supervised machine learning library, it will involve "label" or target training
data.  This data is also provided to `DMatrix` objects and it is kept with the corresponding feature
data.  For example
```julia
using LinearAlgebra

𝒻(x) = 2norm(x)^2 - norm(x)

X = randn(100,2)
y = 𝒻.(eachrow(X))

DMatrix(X, y)  # input data with features X and target y
DMatrix((X, y))  # equivalent (to simplify function arguments)
DMatrix(X, label=y)  # equivalent
(dm = DMatrix(X); XGBoost.setlabel!(dm, y); dm)  # equivalent
```

Training and initialization methods such as `xgboost` and `Booster` can accept feature and label
data together as a tuple
```julia
Booster((X, y))
Booster(DMatrix(X, y)) # equivalent to above
```

Unlike feature data, label data can be extracted after construction of the `DMatrix` with
[`XGBoost.getlabel`](@ref).



## Booster
The [`Booster`](@ref) object holds model data.  They are created with training data.  Internally
this is always a `DMatrix` but arguments will be automatically converted.

### [Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
Keyword arguments to `Booster` are xgboost model parameters.  These are described in detail
[here](https://xgboost.readthedocs.io/en/stable/parameter.html) and should all be passed exactly as
they are described in the main xgbosot documentation (in a few cases such as Greek letters we also
allow unicode equivalents).

!!! note

    The `tree_method` parameter has special handling.  If `nothing`, it will use `libxgboost`
    defaults as per the documentation, unless a GPU array is input in which case it will default to
    `gpu_hist`.  An explicitly set value will override this.

### Training
`Booster` objects can be trained with [`update!`](@ref).
```julia
𝒻(x) = 2norm(x)^2 - norm(x)

X = randn(100,3)
y = 𝒻.(eachrow(X))

bst = Booster((X, y), max_depth=8, η=0.5)

# 20 rounds of training
update!(bst, (X, y), num_round=20)

ŷ = predict(bst, X)

using Statistics
mean(ŷ - y)/std(y)
```

Xgboost expects `Booster`s to be initialized with training data, therefore there is usually no need
to define `Booster` separate from training.  A shorthand for the above, provided by
[`xgboost`](@ref) is
```julia
bst = xgboost((X, y), num_round=20, max_depth=8, η=0.5)

# feature names can also be set here
bst = xgboost((X, y), num_round=20, feature_names=["a", "b"], max_depth=8, η=0.5)
```

Note that `Booster`s can still be boosted with `update!` after they are create with `xgboost` or
otherwise.  For example
```julia
bst = xgboost((X, y), num_round=20)
```
is equivalent to
```julia
bst = xgboost((X, y), num_round=10)
update!(bst, (X, y), num_round=10)
```

### Early Stopping
To help prevent overfitting to the training set, it is helpful to use a validation set to evaluate against to ensure that the XGBoost iterations continue to generalise outside training loss reduction. Early stopping provides a convenient way to automatically stop the
boosting process if it's observed that the generalisation capability of the model does not improve for `k` rounds.

If there is more than one element in watchlist, by default the last element will be used. In this case, you must use an ordered data structure (`OrderedDict`) compared to a standard unordered dictionary otherwise an exception will be generated. There will be
a warning if you want to execute early stopping mechanism (`early_stopping_rounds > 0`) but have provided a watchlist with type `Dict` with
more than 1 element.

Similarly, if there is more than one element in eval_metric, by default the last element will be used.

For example:

```julia
using LinearAlgebra
using OrderedCollections

𝒻(x) = 2norm(x)^2 - norm(x)

X = randn(100,3)
y = 𝒻.(eachrow(X))

dtrain = DMatrix((X, y))

X_valid = randn(50,3)
y_valid = 𝒻.(eachrow(X_valid))

dvalid = DMatrix((X_valid, y_valid))

bst = xgboost(dtrain, num_round = 100, eval_metric = "rmse", watchlist = OrderedDict(["train" => dtrain, "eval" => dvalid]), early_stopping_rounds = 5, max_depth=6, η=0.3)

# get the best iteration and use it for prediction
ŷ = predict(bst, X_valid, ntree_limit = bst.best_iteration)

using Statistics
println("RMSE from model prediction $(round((mean((ŷ - y_valid).^2).^0.5), digits = 8)).")

# we can also retain / use the best score (based on eval_metric) which is stored in the booster
println("Best RMSE from model training $(round((bst.best_score), digits = 8)).")
```