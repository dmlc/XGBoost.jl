module XGBoost

using LinearAlgebra
import SparseArrays
using SparseArrays: SparseMatrixCSC, nnz
import SparseMatricesCSR
using SparseMatricesCSR: SparseMatrixCSR
using AbstractTrees
using OrderedCollections
using JSON3
using Tables
using Statistics: mean, std

using Base: @propagate_inbounds

using Base.Iterators: Stateful, reset!

export DMatrix, Booster
export updateone!, update!, predict, xgboost
export importance, importancetable, importancereport, trees

include("Lib.jl")
using .Lib
using .Lib: DMatrixHandle, BoosterHandle

const LOG_LEVEL_REGEX = r"\[.*\] (\D*): "

function xgblog(s::Cstring)
    s = unsafe_string(s)
    m = match(LOG_LEVEL_REGEX, s)
    if isnothing(m) || isempty(m.captures)
        @info(s)
    elseif m.captures[1] == "WARNING"
        @warn(s)
    else
        @info(s)
    end
end

__init__() = XGBRegisterLogCallback(@cfunction(xgblog, Nothing, (Cstring,)))


include("dmatrix.jl")
include("booster.jl")
include("introspection.jl")
include("show.jl")
include("defaultparams.jl")

if !isdefined(Base, :get_extension)
    include("../ext/XGBoostCUDAExt.jl")
    include("../ext/XGBoostTermExt.jl")
end

end # module XGBoost
