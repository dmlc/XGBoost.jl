```@meta
CurrentModule = XGBoost
```

# Additional Features


## Introspection

### Feature Importance

This package contains a number of methods for inspecting the results of training and displaying the results.

Feature importances can be computed explicitly using [`importance`](@ref)

For a quick and convenient summary one can use [`importancetable`](@ref).  The output of this
function is primarily intended for visual inspection but it is a Tables.jl compatible table so it
can easily be converted to any tabular format.
```julia
bst = xgboost(X, y)

imp = DataFrame(importancetable(bst))
```

XGBoost also supports rich terminal output with [Term.jl](https://github.com/FedeClaudi/Term.jl).
A convenient visualization of this table can also be seen with [`importancereport`](@ref).  These
will use assigned feature names, for example
```julia
julia> df = DataFrame(randn(10,3), ["kirk", "spock", "bones"])
10×3 DataFrame
 Row │ kirk       spock       bones      
     │ Float64    Float64     Float64    
─────┼───────────────────────────────────
   1 │  0.663934  -0.419345   -0.489801
   2 │  1.19064    0.420935   -0.321852
   3 │  0.713867   0.293724    0.0450463
   4 │ -1.3474    -0.402996    1.50831
   5 │ -0.458164   0.0399281  -0.83443
   6 │ -0.277555   0.149485    0.408656
   7 │ -1.79885   -1.1535      0.99213
   8 │ -0.177408  -0.818639    0.280188
   9 │ -1.26053   -1.60734     2.21421
  10 │  0.30378   -0.299256    0.384029

julia> bst = xgboost((df, randn(10)), num_round=10);
[ Info: XGBoost: starting training.
[ Info: [1]     train-rmse:0.57998637329114211
[ Info: [2]     train-rmse:0.48232409595403752
[ Info: [3]     train-rmse:0.40593080843433427
[ Info: [4]     train-rmse:0.34595769369793850
[ Info: [5]     train-rmse:0.29282108263987289
[ Info: [6]     train-rmse:0.24862819795032731
[ Info: [7]     train-rmse:0.21094418685218519
[ Info: [8]     train-rmse:0.17903024616536045
[ Info: [9]     train-rmse:0.15198720040980171
[ Info: [10]    train-rmse:0.12906074380448287
[ Info: Training rounds complete.

julia> using Term; Panel(bst)
╭──── XGBoost.Booster ─────────────────────────────────────────────────────────────────╮
│  Features: ["kirk", "spock", "bones"]                                                │
│                                                                                      │
│          Parameter          Value                                                    │
│   ─────────────────────────────────                                                  │
│     validate_parameters     true                                                     │
│                                                                                      │
╰──── boosted rounds: 10 ──────────────────────────────────────────────────────────────╯

julia> importancereport(bst)
╭───────────┬─────────────┬──────────┬───────────┬──────────────┬───────────────╮
│  feature  │    gain     │  weight  │   cover   │  total_gain  │  total_cover  │
├───────────┼─────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "bones"  │  0.358836   │   15.0   │  8.53333  │   5.38254    │     128.0     │
├───────────┼─────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "spock"  │  0.157437   │   16.0   │   4.75    │   2.51899    │     76.0      │
├───────────┼─────────────┼──────────┼───────────┼──────────────┼───────────────┤
│  "kirk"   │  0.0128546  │   34.0   │  2.91176  │   0.437056   │     99.0      │
╰───────────┴─────────────┴──────────┴───────────┴──────────────┴───────────────╯
```

### Tree Inspection
The trees of a model belonging to a `Booster` can retrieved and directly inspected with
[`trees`](@ref) which returns an array of [`Node`](@ref) objects each representing the model
from a single round of boosting.

Tree objects satisfy the [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)
interface.

```julia
julia> ts = trees(bst)
10-element Vector{XGBoost.Node}:
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="bones")

julia> using Term; Panel(ts[1])
╭──── XGBoost.Node (id=0, depth=0) ────────────────────────────────────────────────────╮
│                                                                                      │
│     split_condition     yes     no     nmissing        gain        cover             │
│   ────────────────────────────────────────────────────────────────────────           │
│       0.396342576        1      2         1         1.86042714     10.0              │
│                                                                                      │
│   XGBoost Tree (from this node)                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                     │
│                │                                                                     │
│                ├── bones < 0.396                                                     │
│                │   ├── bones < 0.332: XGBoost.Node(leaf=-0.159539297)                │
│                │   └── bones ≥ 0.332: XGBoost.Node(leaf=-0.0306737479)               │
│                └── bones ≥ 0.396                                                     │
│                    ├── spock < -0.778                                                │
│                    │   ├── kirk < -1.53: XGBoost.Node(leaf=-0.0544514731)            │
│                    │   └── kirk ≥ -1.53: XGBoost.Node(leaf=0.00967349485)            │
│                    └── spock ≥ -0.778                                                │
│                        ├── kirk < -0.812: XGBoost.Node(leaf=0.0550933369)            │
│                        └── kirk ≥ -0.812: XGBoost.Node(leaf=0.228843644)             │
╰──── 2 children ──────────────────────────────────────────────────────────────────────╯

julia> using AbstractTrees; children(ts[1])
2-element Vector{XGBoost.Node}:
 XGBoost.Node(split_feature="bones")
 XGBoost.Node(split_feature="spock")
```

## Setting a Custom Objective Function
Xgboost uses a second order approximation, so to provide a custom objective functoin first and
second order derivatives must be provided, see the docstring of [`updateone!`](@ref) for more
details.

While the derivatives can be provided manually, it is also easy to use a calculus package to compute
them and supply them to xgboost.  Julia is notorious for having a large number of
auto-differentiation packages.  To provide an example we will use one of the most popular such
packages [Zygote.jl](https://github.com/FluxML/Zygote.jl)
```julia
using Zygote, XGBoost

# we use squared error loss to demonstrate
ℓ(ŷ, y) = (ŷ - y)^2

# we will try to fit this function
𝒻(x) = 2norm(x)^2 - norm(x)
X = randn(100, 2)
y = 𝒻.(eachrow(X))

# this is the (scalar) first derivative of the loss
ℓ′ = (ŷ, y) -> gradient(ζ -> ℓ(ζ, y), ŷ)[1]

# this is the (scalar) second derivative of the losss
ℓ″ = (ŷ, y) -> gradient(ζ -> ℓ′(ζ, y), ŷ)[1]

# the derivatives are the non-keyword arguments after the data,
# keyword arguments can be provided as usual
bst = xgboost((X, y), ℓ′, ℓ″, max_depth=8)
```

## Caching Data From External Memory
Xgboost can be used to cache memory from external memory on disk, see
[here](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html).  In the Julia
wrapper this is facilitated by allowing a `DMatrix` to be constructed from any Julia iterator with
[`fromiterator`](@ref).  The resulting `DMatrix` holds references to cache files which will have
been created on disk.  For example
```julia
Xy = [(X=randn(10,4), y=randn(10)) for i ∈ 1:5]
dm = XGBoost.fromiterator(DMatrix, Xy, cache_prefix=pwd())
```
will create a `DMatrix` that will use the present working directory to store cache files (if
`cache_prefix` is not set this will be in `/tmp`).  Objects returned by the supplied iterator must
have `Symbol` keys which can be used to supply arguments to `DMatrix` with `:X` being the key for
the main matrix and `:y` being the key for labels (typically a `NamedTuple` or a
`Dict{Symbol,Any}`).


## Default Parameters
This wrapper can provide reasonable defaults for the following
- [`regression`](@ref)
- [`countregression`](@ref)
- [`classification`](@ref)
- [`randomforest`](@ref)

Each of these merely returns a `NamedTuple` which can be used to supply keyword arguments to
`Booster` or `xgboost`.  For example
```julia
xgboost(X, y, 1; countregression()..., randomforest()..., num_parallel_tree=12)
```
will fit a random forest according to a Poisson likelihood fit with 12 trees.


## GPU Support
XGBoost supports GPU-assisted training on Nvidia GPU's with CUDA via
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).  To utilize the GPU, one has to load CUDA and construct a
`DMatrix` object from GPU arrays.  There are two ways of doing this:
- Pass a `CuArray` as the training matrix (conventionally `X`, the first argument to `DMatrix`).
- Pass a table with *all* columns as `CuVector`s.

You can check whether a `DMatrix` can use the GPU with [`XGBoost.isgpu`](@ref).

The target or label data does not need to be a `CuArray`.

It is not necessary to create an explicit `DMatrix` to use GPU features, one can pass the data
normally directly to `xgboost` or `Booster`, as long as that data consists of `CuArray`s.

!!! note

    The `tree_method` parameter to `Booster` has special handling.  If `nothing`, it will use `libxgboost`
    defaults as per the documentation, unless a GPU array is given in which case it will default to
    `gpu_hist`.  An explicitly set value will override this.

### Example
```julia
using CUDA

X = cu(randn(1000, 3))
y = randn(1000)

dm = DMatrix(X, y)
XGBoost.isgpu(dm)  # true

X = (x1=cu(randn(1000)), x2=cu(randn(1000)))
dm = DMatrix(X, y)
XGBoost.isgpu(dm)  # true

xgboost((X, y), num_rounds=10)  # no need to use `DMatrix`
```
