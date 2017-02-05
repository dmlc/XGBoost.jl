# TODO: Get rid of type-conversions where possible
# TODO: Relax type requirements for exported functions

type DMatrix
    handle::DMatrixHandle


    function DMatrix(handle::DMatrixHandle)
        dmat = new(handle)
        finalizer(dmat, JLFree)
        return dmat
    end


    function DMatrix{K<:Real,V<:Integer}(data::SparseMatrixCSC{K,V};
        label = nothing, weight = nothing, transposed::Bool = false)

        handle = transposed ? XGDMatrixCreateFromCSCT(data) : XGDMatrixCreateFromCSC(data)
        dmat = new(handle)
        finalizer(dmat, JLFree)

        if typeof(label) != Void
            XGDMatrixSetFloatInfo(dmat.handle, "label", label, length(label))
        end
        if typeof(weight) != Void
            XGDMatrixSetFloatInfo(dmat.handle, "weight", weight, length(weight))
        end

        return dmat
    end


    function DMatrix{T<:Real}(data::Matrix{T};
        label = nothing, missing::Real = NaN32,
        weight = nothing, transposed::Bool = false)

        if !transposed
            handle = XGDMatrixCreateFromMat(data, missing)
        else
            handle = XGDMatrixCreateFromMatT(data, missing)
        end

        dmat = new(handle)
        finalizer(dmat, JLFree)

        if typeof(label) != Void
            XGDMatrixSetFloatInfo(dmat.handle, "label", label, length(label))
        end
        if typeof(weight) != Void
            XGDMatrixSetFloatInfo(dmat.handle, "weight", weight, length(weight))
        end

        return dmat
    end


    function DMatrix(fname::String;
                     silent = false)
        handle = XGDMatrixCreateFromFile(fname, silent)
        dmat = new(handle)
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
    return XGDMatrixGetFloatInfo(dmat.handle, "base_margin")
end


function get_float_info(dmat::DMatrix, field::String)
    return XGDMatrixGetFloatInfo(dmat.handle, field)
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
    XGDMatrixSaveBinary(dmat.handle, fname, silent)
    return nothing
end


function set_base_margin{T<:Real}(dmat::DMatrix, margin::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "base_margin", margin, length(margin))
    return nothing
end


function set_float_info{T<:Real}(dmat::DMatrix, field::String, data::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, field, data, length(data))
    return nothing
end


function set_group{T<:Integer}(dmat::DMatrix, group::Vector{T})
    XGDMatrixSetGroup(dmat.handle, group, length(group))
    return nothing
end


function set_label{T<:Real}(dmat::DMatrix, label::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "label", label, length(label))
    return nothing
end


function set_uint_info{T<:Integer}(dmat::DMatrix, field::String, data::Vector{T})
    XGDMatrixSetUIntInfo(dmat.handle, field, data, length(data))
    return nothing
end


function set_weight{T<:Real}(dmat::DMatrix, weight::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "weight", weight, length(weight))
    return nothing
end


function slice{T<:Integer}(dmat::DMatrix, rindex::Vector{T})
    handle = XGDMatrixSliceDMatrix(dmat.handle, rindex - 1, length(rindex))
    return DMatrix(handle)
end


function makeDMatrix(data, label)
    # running converts
    if typeof(data) != DMatrix
        if typeof(data) == String
            if label != nothing
                warning("label will be ignored when data is a file")
            end
            return DMatrix(data)
        else
            if label == nothing
                error("label argument must be present for training, unless you pass in a DMatrix")
            end
            return DMatrix(data, label = label)
        end
    else
        return data
    end
end
