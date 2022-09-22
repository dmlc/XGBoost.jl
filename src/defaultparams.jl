

"""
    regression(;kw...)

Default parameters for performing a regression.  Returns a named tuple that can be used to supply
arguments in the usual way

## Example
```julia
using XGBoost: regression

regression()  # will merely return default parameters as named tuple

xgboost(X, y, 10; regression(max_depth=8)...)
xgboost(X, y, 10; regression()..., max_depth=8)
```
"""
regression(;kw...) = (objective="reg:squarederror",
                      eval_metric="rmse",
                      kw...
                     )

"""
    countregression(;kw...)

Default parameters for performing a regression on a Poisson-distributed variable.
"""
countregression(;kw...) = (objective="count:poisson",
                           eval_metric="rmse",
                           kw...
                          )

"""
    classification(;kw...)

Default parameters for performing a classification.
"""
classification(;kw...) = (objective="multi:softmax",
                          eval_metric="mlogloss",
                          kw...
                         )

"""
    randomforest(;kw...)

Default parameters for training as a random forest.  Note that a conventional random forest would involve
using these parameters with exactly *1* round of boosting, however there is nothing stopping you from boosting
`n` random forests.

Parameters that are particularly relevant to random forests are:
- `num_parallel_tree`: number of trees in the forest.
- `subsample`: Sample fraction of data (occurs once per boosting iteration).
- `colsample_bynode`: Sampling fraction of data on node splits.
- `η`: Learning rate, when set to `1` there is no shrinking of updates.

See [here](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html) for more details.

## Examples
```julia
using XGBoost: regression, randomforest

xgboost(X, y, 1; regression()..., randomforest()...)
```
"""
randomforest(;kw...) = (booster="gbtree",
                        subsample=0.9,
                        colsample_bynode=0.8,
                        num_parallel_tree=10,
                        η=1.0,
                        kw...
                       )

