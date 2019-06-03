rabit_finalize() = ccall((:RabitFinalize, _jl_libxgboost), Nothing, ())
function rabit_init()
    argv = Ref{Ptr{Nothing}}()
    ccall((:RabitInit, _jl_libxgboost), Nothing, (Cint, Ref{Ptr{Nothing}}), 0, argv)
end

function rabit_is_distributed()
    ret = ccall((:RabitIsDistributed, _jl_libxgboost), Cint, ())
    return (ret != Cint(0))
end

rabit_get_rank() = ccall((:RabitGetRank, _jl_libxgboost), Cint, ())

rabit_get_world_size() = ccall((:RabitGetWorldSize, _jl_libxgboost), Cint, ())

rabit_get_version_number() = ccall((:RabitVersionNumber, _jl_libxgboost), Cint, ())
