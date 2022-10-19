# see https://juliainterop.github.io/Clang.jl/stable/generator/
using Clang
using Clang.Generators
using XGBoost_jll

cd(@__DIR__)

include_dir = joinpath(XGBoost_jll.artifact_dir,"include","xgboost")

opts = load_options(joinpath(@__DIR__,"generator.toml"))

args = get_default_args()
push!(args, "-I$include_dir")

ctx = create_context([joinpath(include_dir,"c_api.h")], args, opts)

build!(ctx)
