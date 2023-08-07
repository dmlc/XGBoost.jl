```@meta
CurrentModule = XGBoost
```

# API

```@index
```

## Data Input
```@docs
DMatrix
load(::Type{DMatrix}, ::AbstractString)
save(::DMatrix, ::AbstractString)
setfeaturenames!
getfeaturenames
setinfo!
setinfos!
setlabel!
getinfo
slice
nrows
ncols
size(::DMatrix)
isgpu
getlabel
getweights
setfeatureinfo!
getfeatureinfo
setproxy!
DataIterator
fromiterator
```

## Training and Prediction
```@docs
xgboost
Booster
updateone!
update!
predict
predict_nocopy
setparam!
setparams!
getnrounds
load!
load(::Type{Booster}, ::AbstractString)
save(::Booster, ::AbstractString)
serialize
nfeatures
deserialize!
deserialize
```

## Introspection
```@docs
trees
importancetable
importance
importancereport
Node
dump
dumpraw
```

## Default Parameters
```@docs
regression
countregression
classification
randomforest
```
