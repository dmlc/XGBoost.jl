# TODO: Get rid of type-conversions where possible
# TODO: Relaxe type requirements for exported functions

type DMatrix
    handle::DMatrixHandle

    function DMatrix(handle::DMatrixHandle)
        dmat = new(handle)
        finalizer(dmat, JLFree)
        return dmat
    end

    function DMatrix(fname::String;
                     silent = false)
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


# function feature_names(dmat::DMatrix)
# function feature_types(dmat::DMatrix)


function get_base_margin(dmat::DMatrix)
    XGDMatrixGetFloatInfo(dmat.handle, "base_margin")
end


function get_float_info(dmat::DMatrix, field::String)
    XGDMatrixGetFloatInfo(dmat.handle, field)
end


function get_label(dmat::DMatrix)
    return XGDMatrixGetFloatInfo(dmat.handle, "label")
end


function get_uint_info(dmat::DMatrix, field::String)
    return XGDMatrixGetUIntInfo(dmat.handle, field)
end


function get_weight(dmat::DMatrix)
    return XGDMatrixGetFloatInfo(dmat.handle, "weight")
end


function num_col(dmat::DMatrix)
    return XGDMatrixNumCol(dmat.handle)
end


function num_row(dmat::DMatrix)
    return XGDMatrixNumRow(dmat.handle)
end


function save_binary(dmat::DMatrix, fname::String;
                     silent = true)
    XGDMatrixSaveBinary(dmat.handle, fname, convert(Int32, silent))
    return nothing
end


function set_base_margin(dmat::DMatrix, margin::Vector{Float32})
    XGDMatrixSetFloatInfo(dmat.handle, "base_margin", margin, convert(UInt64, length(margin)))
    return nothing
end


function set_float_info(dmat::DMatrix, field::String, data::Array{Float32})
    XGDMatrixSetFloatInfo(dmat.handle, field, data, convert(UInt64, length(data)))
    return nothing
end


function set_group(dmat::DMatrix, group::Vector{UInt32})
    XGDMatrixSetGroup(dmat.handle, group, convert(UInt64, length(group)))
    return nothing
end


function set_label(dmat::DMatrix, label::Vector{Float32})
    XGDMatrixSetFloatInfo(dmat.handle, "label", label, convert(UInt64, length(label)))
    return nothing
end


function set_uint_info(dmat::DMatrix, field::String, data::Array{UInt32})
    XGDMatrixSetUIntInfo(handle.handle, field, data, convert(UInt64, length(data)))
    return nothing
end


function set_weight(dmat::DMatrix, weight::Vector{Float32})
    XGDMatrixSetFloatInfo(dmat.handle, "weight", weight, convert(UInt64, length(weight)))
    return nothing
end


function slice{T<:Integer}(dmat::DMatrix, rindex::Vector{T})
    handle = XGDMatrixSliceDMatrix(dmat.handle, convert(Vector{Int32}, rindex - 1),
                                   convert(UInt64, length(rindex)))
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
