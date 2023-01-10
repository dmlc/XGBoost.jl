# XGBoost.jl

[![Build Status](https://github.com/dmlc/XGBoost.jl/workflows/CI/badge.svg)](https://github.com/dmlc/XGBoost.jl/actions)
[![Latest Version](https://juliahub.com/docs/XGBoost/version.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/)
[![Pkg Eval](https://juliahub.com/docs/XGBoost/pkgeval.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/)
[![Dependents](https://juliahub.com/docs/XGBoost/deps.svg)](https://juliahub.com/ui/Packages/XGBoost/rSeEh/?t=2)
[![docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=julia)](https://dmlc.github.io/XGBoost.jl/)

eXtreme Gradient Boosting in Julia.

## Abstract
This package is a Julia interface of [XGBoost](https://github.com/dmlc/xgboost). 
It is an efficient and scalable implementation of distributed gradient boosting
framework. The package includes efficient linear model solver and tree learning algorithms. The
library is parallelized using OpenMP, and it can be more than 10 times faster than some existing
gradient boosting packages. It supports various objective functions, including regression,
classification and ranking. The package is also made to be extensible, so that users are also
allowed to define their own objectives easily.

See the [documentation](https://dmlc.github.io/XGBoost.jl/) for more information.

## Installation
```julia
] add XGBoost
```
This package uses [`xgboost_jll`](https://github.com/JuliaBinaryWrappers/XGBoost_jll.jl) to package
the `xgboost` binaries (will be installed automatically).

## Preview
![](/assets/xgboost_demo_1.jpg)

