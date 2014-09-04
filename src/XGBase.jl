module XGBoost


### DMatrix ###
type DMatrix
    handle::Ptr{Void}
    function DMatrix(fname, slient)
        handle = ccall((:XGDMatrixCreateFromFile,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                    Ptr{Void},
                    (Ptr{Uint8}, Int),
                    fname, slient)
        new(handle)
    end
end

function XGDMatrixFree(dmat::DMatrix)
    ccall((:XGDMatrixFree,
        "../xgboost/wrapper/libxgboostwrapper.so"),
        Void,
        (Ptr{Void}, ),
        dmat.handle)
end

### Booster ###
type Booster
    handle::Ptr{Void}
    function Booster(dmats::Array{Ptr{None}, 1}, len::Int64)
        handle = ccall((:XGBoosterCreate,
                    "../xgboost/wrapper/libxgboostwrapper.so"),
                    Ptr{Void},
                    (Ptr{Ptr{Void}}, Culong),
                    dmats, len)
        new(handle)
    end
end

function XGBoosterFree(bst::Booster)
    ccall((:XGBoosterFree,
        "../xgboost/wrapper/libxgboostwrapper.so"),
        Void,
        (Ptr{Void}, ),
        bst.handle)
end

function XGBoosterSetParam(bst::Booster, key::ASCIIString, value::ASCIIString)
    ccall((:XGBoosterSetParam,
            "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
          bst.handle, key, value)
end

function XGBoosterUpdateOneIter(bst::Booster, iter::Int64, dtrain::DMatrix)
    ccall((:XGBoosterUpdateOneIter,
           "../xgboost/wrapper/libxgboostwrapper.so"),
          Void,
          (Ptr{Void}, Int32, Ptr{Void}),
          bst.handle, iter, dtrain.handle)
end


