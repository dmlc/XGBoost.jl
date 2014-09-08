XGBoost.jl
==========

eXtreme Gradient Boosting interface in Julia

## Abstruct

XGBoost is an optimized general purpose gradient boosting library. The library is parallelized using OpenMP. It implements machine learning algorithm under gradient boosting framework, including generalized linear model and gradient boosted regression tree.

Among kaggle competitions, XGBoost has shown strong advantages compare to other boosting implmentations.

## Installation

TODO


The `XGBoost` package also depends on the `BinDeps`

## Data in XGBoost
All data should be transformed to `DMatrix` like
```julia
dtrain = DMatrix("train.svm.txt")
dtest = DMatrix("test.svm.txt")
```

`DMatrix` support to be built directly from
- `libSVM` txt format file
- `XGBoost` buffer file
- Julia `Array{Real, 2}`
- Julia `SparseMatrixCSC{Real, Int}`

Check [ demo/basic_walkthrough.jl](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/basic_walkthrough.jl) for all usage

## Model in XGBoost
Booster



