rabit_finalize() = ccall((:RabitFinalize, libxgboost), Nothing, ())
function rabit_init()
    argv = Ref{Ptr{Nothing}}()
    ccall((:RabitInit, libxgboost), Nothing, (Cint, Ref{Ptr{Nothing}}), 0, argv)
end

function rabit_is_distributed()
    ret = ccall((:RabitIsDistributed, libxgboost), Cint, ())
    return (ret != Cint(0))
end

rabit_get_rank() = ccall((:RabitGetRank, libxgboost), Cint, ())

rabit_get_world_size() = ccall((:RabitGetWorldSize, libxgboost), Cint, ())

rabit_get_version_number() = ccall((:RabitVersionNumber, libxgboost), Cint, ())
