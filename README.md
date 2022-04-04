XGBoost.jl
==========

[![Build Status](https://github.com/dmlc/XGBoost.jl/workflows/CI/badge.svg)](https://github.com/dmlc/XGBoost.jl/actions)
[![Latest Version](https://juliahub.com/docs/XGBoost/version.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/)
[![Pkg Eval](https://juliahub.com/docs/XGBoost/pkgeval.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/)
[![Dependents](https://juliahub.com/docs/XGBoost/deps.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/?t=2)

eXtreme Gradient Boosting in Julia

## Abstract
This package is a Julia interface of [XGBoost](https://github.com/dmlc/xgboost). 
It is an efficient and scalable implementation of distributed gradient boosting
framework. The package includes efficient linear model solver and tree learning algorithms. The
library is parallelized using OpenMP, and it can be more than 10 times faster than some existing
gradient boosting packages. It supports various objective functions, including regression,
classification and ranking. The package is also made to be extensible, so that users are also
allowed to define their own objectives easily.

## Features
* Sparse feature format, it allows easy handling of missing values, and improve computation
    efficiency.
* Advanced features, such as customized loss function, cross validation, see [demo folder](demo)
    for walkthrough examples.

## Installation
```julia
] add XGBoost
```
or
```julia
] develop "https://github.com/dmlc/XGBoost.jl.git"
] build XGBoost
```

By default, the package installs prebuilt binaries for XGBoost `v0.82.0` on Linux, MacOS and Windows. Only the linux version is built with OpenMP. 


## Minimal examples

To show how XGBoost works, here is an example of dataset Mushroom

- Prepare Data

XGBoost support Julia ```Array```, ```SparseMatrixCSC```, libSVM format text and XGBoost binary
file as input. Here is an example of Mushroom classification. This example will use the function
```readlibsvm``` in [basic_walkthrough.jl](demo/basic_walkthrough.jl#L5). This function load libsvm
format text into Julia dense matrix.

```julia
using XGBoost

train_X, train_Y = readlibsvm("data/agaricus.txt.train", (6513, 126))
test_X, test_Y = readlibsvm("data/agaricus.txt.test", (1611, 126))

```

- Fit Model
```julia
num_round = 2
bst = xgboost(train_X, num_round, label = train_Y, eta = 1, max_depth = 2)
```

## Predict
```julia
pred = predict(bst, test_X)
print("test-error=", sum((pred .> 0.5) .!= test_Y) / float(size(pred)[1]), "\n")
```

## Cross-Validation
```julia
nfold = 5
param = ["max_depth" => 2,
         "eta" => 1,
         "objective" => "binary:logistic"]
metrics = ["auc"]
nfold_cv(train_X, num_round, nfold, label = train_Y, param = param, metrics = metrics)
```

## Feature Walkthrough
Check [demo](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/)

- [Basic walkthrough of features](demo/basic_walkthrough.jl)
- [Customize loss function, and evaluation metric](demo/custom_objective.jl)
- [Boosting from existing prediction](demo/boost_from_prediction.jl)
- [Predicting using first n trees](demo/predict_first_ntree.jl)
- [Generalized Linear Model](demo/generalized_linear_model.jl)
- [Cross validation](demo/cross_validation.jl)


## Model Parameter Setting
Check [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/parameter.html)
