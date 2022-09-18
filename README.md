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

