
struct XGBoostError <: Exception
    caller
    message::String
end

function Base.showerror(io::IO, err::XGBoostError)
    println(io, "XGBoostError: (caller: $(string(err.caller)))")
    print(io, err.message)
end

"""
    xgbcall(𝒻, a...)

Call xgboost library function `𝒻` on arguments `a` properly handling errors.
"""
function xgbcall(𝒻, a...)
    err = 𝒻(a...)
    if err ≠ 0
        msg = unsafe_string(XGBGetLastError())
        throw(XGBoostError(𝒻, msg))
    end
    err
end

export XGBoostError, xgbcall
