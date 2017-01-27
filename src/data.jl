type DMatrix
    handle::DMatrixHandle

    function DMatrix(handle::DMatrixHandle)
        dmat = new(handle)
        finalizer(dmat, JLFree)
        return dmat
    end

    function DMatrix(fname::String; silent = false)
        handle = XGDMatrixCreateFromFile(fname, convert(Int32, silent))
        dmat = new(handle)
        finalizer(dmat, JLFree)
        return dmat
    end

    function DMatrix{K<:Real, V<:Integer}(data::SparseMatrixCSC{K,V}, transposed::Bool = false;
                                          kwargs...)
        handle = transposed ? XGDMatrixCreateFromCSCT(data) : XGDMatrixCreateFromCSC(data)
        dmat = new(handle)
        for itm in kwargs
            setinfo(dmat, string(itm[1]), itm[2])
        end
        finalizer(dmat, JLFree)
        return dmat
    end

    function DMatrix{T<:Real}(data::Matrix{T}, transposed::Bool = false, missing = NaN32;
                              kwargs...)
        if !transposed
            handle = XGDMatrixCreateFromMat(convert(Matrix{Float32}, data),
                                            convert(Float32, missing))
        else
            handle = XGDMatrixCreateFromMatT(convert(Matrix{Float32}, data),
                                             convert(Float32, missing))
        end

        dmat = new(handle)
        for itm in kwargs
            setinfo(dmat, string(itm[1]), itm[2])
        end

        finalizer(dmat, JLFree)
        return dmat
    end

    function JLFree(dmat::DMatrix)
        XGDMatrixFree(dmat.handle)
    end
end


function get_info(dmat::DMatrix, field::String)
    XGDMatrixGetFloatInfo(dmat.handle, field)
end


function setinfo{T<:Real}(dmat::DMatrix, name::String, array::Vector{T})
    if name == "label" || name == "weight" || name == "base_margin"
        XGDMatrixSetFloatInfo(dmat.handle, name,
                              convert(Vector{Float32}, array),
                              convert(UInt64, length(array)))
    elseif name == "group"
        XGDMatrixSetGroup(dmat.handle,
                          convert(Vector{UInt32}, array),
                          convert(UInt64, length(array)))
    else
        error("unknown information name")
    end
end


function save(dmat::DMatrix, fname::String; silent = true)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, silent))
end


### slice ###
function slice{T<:Integer}(dmat::DMatrix, idxset::Vector{T})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Vector{Int32}, idxset - 1),
                                   convert(UInt64, size(idxset)[1]))
    return DMatrix(handle)
end


function makeDMatrix(data, label)
    # running converts
    if typeof(data) != DMatrix
        if typeof(data) == String
            if label != Union{}
                warning("label will be ignored when data is a file")
            end
            return DMatrix(data)
        else
            if label == Union{}
                error("label argument must be present for training, unless you pass in a DMatrix")
            end
            return DMatrix(data, label = label)
        end
    else
        return data
    end
end
