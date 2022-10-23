using XGBoost
using Documenter

DocMeta.setdocmeta!(XGBoost, :DocTestSetup, :(using XGBoost); recursive=true)

makedocs(;
    modules=[XGBoost],
    repo="https://github.com/dmlc/XGBoost.jl/blob/{commit}{path}#{line}",
    sitename="XGBoost.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dmlc.github.io/XGBoost.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Additional Features" => "features.md",
        "API" => "api.md",
       ],
   )

deploydocs(repo="github.com/dmlc/XGBoost.jl.git", devbranch="master")
