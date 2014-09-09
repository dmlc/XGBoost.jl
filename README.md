XGBoost.jl
==========

eXtreme Gradient Boosting Package in Julia

## Abstract

This package is a Julia interface of [XGBoost](https://github.com/tqchen/xgboost), 
whieh is short for eXtreme gradient Gradient Boosting.  It is an efficient and scalable implementation of
gradient boosting framework.The package includes efficient linear model
solver and tree learning algorithms. The library is parallelized using OpenMP,
and it can be more than 10 times faster some of than existing gradient boosting packages.
It supports various objective functions, including regression, classification and ranking.
The package is also made to be extensible, so that users are also allowed to define their own objectives easily.

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
- Julia `SparseMatrixCSC{K, V}` where ```K<:Real``` and ```V<:Int```

Check [ demo/basic_walkthrough.jl](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/basic_walkthrough.jl) for all usage

## Model in XGBoost
```Booster``` is the model struct. To get ```Booster``` object, you can load from saved model file or use method ```xgboost`` to fit a new model

## Fit model
To fit a model, use the function ```bst = xgboost(dtrain, nrounds, param=param, watchlist=watchlist, obj=obj, feval=feval)``` where
- ```dtrain``` training data in ```DMatrix```, required
- ```nrounds``` training epoch in ```Integer```, required
- ```param``` parameter pairs for XGBoost in ```Dict``` or ```Array{(ASCIIString, Any), 1}```, optional
- ```watchlist``` data need to be evaulated in ```Array{(DMatrix, ASCIIString), 1}```, optional
- ```obj``` customized objective function to be boosted, optional
- ```feval``` customized evaluation function, optional

Check [ demo](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/) for all usage

## Prediction using fitted model
To do prediction, use the function ```preds = predict(bst, dmat, output_margin=output_margin, ntree_limit=ntree_limit)``` where
- ```bst``` fitted model ```Booster``` object, required
- ```dmat``` data to be predicted in ```DMatrix```, required
- ```output_margin``` whether to change the output to margin probability, ```Bool```, optional
- ```ntree_limit``` number of trees used in prediction, ```Integer```, optional

Check [ demo](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/) for all usage

## Other useful function

To save ```Booster``` object into binary file, use the function ```save(bst, fname)```, where
- ```bst``` fitted model ```Booster``` object, required
- ```fname``` output file name in ```ASCIIString```, required
 
To save ```DMatrix``` object into binary file, use the function ```save(dmat, fname)```, where
- ```dmat``` data in ```DMatrix``` object, required
- ```fname``` output file name in ```ASCIIString```, required

To dump ```Booster``` model into readable txt file, use the function ```dump_model(bst, fname, fmap=fmap)``` where
- ```bst``` fitted model ```Booster``` object, required
- ```fname``` output file name in ```ASCIIString```, required
- ```fmap``` file path to feature map file, optional

To slice ```DMatrix```, use the function ```dnew = slice(dmat, idxset)``` where
- ```dmat``` data in ```DMatrix``` object, required
- ```idxset``` id of index to be selected in ```Array{T, 1}```, required

To get ```label```, ```weight```, ```base_margin``` and ```group``` from ```DMatrix```, use function ```get_info(dmat, field)``` where 
- ```dmat``` data in ```DMatrix``` object, required
- ```field``` ```ASCIIString``` of ```label```, ```weight```, ```base_margin``` and ```group```

To set ```label```, ```weight```, ```base_margin``` and ```group``` to ```DMatrix```, use function ```set_info(dmat, field, array)``` where 
- ```dmat``` data in ```DMatrix``` object, required
- ```field``` ```ASCIIString``` of ```label```, ```weight```, ```base_margin``` and ```group```
- ```array```meta info in ```Array{T, 1}``` to be set into dmat

To do cross validation, use the function ```nfold_cv(param, dtrain, num_boost_round, nfold, metrics=metrics, obj=obj, feval=feval, fpreproc=fpreproc, show_stdv=show_stdv, seed=seed)``` where
- ```param``` parameter pairs for XGBoost in ```Dict``` or ```Array{(ASCIIString, Any), 1}```, required
- ```dtrain``` training data in ```DMatrix```, required
- ```num_boost_round``` training epoch in ```Integer```, required
- ```nfold``` fold used for cv in ```Integer```, required
- ```metrics``` metrics used in cv in ```Array{ASCIIString, 1}```, optional
- ```obj``` customized objective function to be boosted, optional
- ```feval``` customized evaluation function, optional
- ```fpreproc``` customized preprocessing function, optional
- ```show_stdv``` whether show std value for result, optional
- ```seed``` random seed in ```Integer```

Check [demo](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/) for all usage
Check [XGBoost Wiki](https://github.com/tqchen/xgboost/wiki) for param setting


