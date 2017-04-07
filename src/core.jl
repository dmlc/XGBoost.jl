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
    return Int(XGDMatrixNumCol(dmat.handle))
end


function num_row(dmat::DMatrix)
    return Int(XGDMatrixNumRow(dmat.handle))
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


# TODO: Deprecate
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


type Booster
    handle::BoosterHandle

    function Booster(;
                     cachelist::Vector{DMatrix} = DMatrix[], model_file::String = "")
        handle = XGBoosterCreate([itm.handle for itm in cachelist], length(cachelist))
        if model_file != ""
            XGBoosterLoadModel(handle, model_file)
        end
        bst = new(handle)
        finalizer(bst, JLFree)
        return bst
    end

    function JLFree(bst::Booster)
        XGBoosterFree(bst.handle)
    end
end


function attr(bst::Booster, attr::String)
    return XGBoosterGetAttr(bst.handle, attr)
end


function attributes(bst::Booster)
    names = XGBoosterGetAttrNames(bst)
    result = Dict{String,String}()
    for name in names
        result[name] = XGBoosterGetAttr(bst.handle, name)
    end
    return result
end


function boost(bst::Booster, dtrain::DMatrix, grad::Vector{Float32}, hess::Vector{Float32})
    @assert size(grad) == size(hess)
    XGBoosterBoostOneIter(bst.handle, dtrain.handle, grad, hess, length(hess))
    return nothing
end


function copy(bst::Booster)
    return Booster(model_file = save_raw(bst))
end


function dump_model(bst::Booster, fout::String;
                    fmap::String = "", with_stats::Bool = false)
    model = XGBoosterDumpModel(bst.handle, fmap, with_stats)
    file = open(fout, "w")
    try
        for i in 1:length(model)
            @printf(file, "booster[%d]:\n", i)
            @printf(file, "%s", unsafe_string(model[i]))
        end
    finally
        close(file)
    end
    return nothing
end


function eval(bst::Booster, data::DMatrix;
              name::String = "eval", iteration::Int = 0)
    return eval_set(bst, [(data, name)], iteration)
end


function eval_set(bst::Booster, evals::Vector{Tuple{DMatrix,String}}, iteration::Integer;
                  feval::Union{Function,Void} = nothing)
    dmats = DMatrix[eval[1] for eval in evals]
    evnames = String[eval[2] for eval in evals]

    result = ""
    if typeof(feval) == Function
        result *= @sprintf("[%d]", iteration)
        for eval_idx in 1:length(dmats)
            pred = predict(bst, dmats[eval_idx])
            name, val = feval(pred, dmats[eval_idx])
            result *= @sprintf("\t%s-%s:%f", evnames[eval_idx], name, val)
        end
        result *= @sprintf("\n")
    else
        result *= @sprintf("%s\n", XGBoosterEvalOneIter(bst.handle, iteration,
                                                        [dmat.handle for dmat in dmats], evnames,
                                                        length(dmats)))
    end
    return result
end


function get_dump(bst::Booster;
                  fmap = "", with_stats = false)
    raw_dump = XGBoosterDumpModel(bst.handle, fmap, with_stats)
    model = [unsafe_string(ptr) for ptr in raw_dump]
    return return model
end


# function get_fscore(bst::Booster; fmap = "")
# function get_score(bst::Booster; fmap = "", importance_type = "weight")
# function get_split_value_histogram(bst::Booster, feature::String; fmap = "", bins = nothing)


function load_model(fname::String)
    bst = Booster()
    XGBoosterLoadModel(bst.handle, fname)
    return bst
end


function load_model(fname::Vector{UInt8})
    bst = Booster()
    len = length(fname)
    XGBoosterLoadModelFromBuffer(bst.handle, fname, len)
    return bst
end


# function load_rabit_checkpoint()


function predict(bst::Booster, data::DMatrix;
                 output_margin::Bool = false, ntree_limit::Integer = 0)
    option_mask = 0x00
    if output_margin
        option_mask |= 0x01
    end

    return XGBoosterPredict(bst.handle, data.handle, option_mask, ntree_limit)
end


function predict_leaf(bst::Booster, data::DMatrix;
                      ntree_limit::Integer = 0)
    option_mask = 0x02
    pred = XGBoosterPredict(bst.handle, data.handle, option_mask, ntree_limit)
    n_row = num_row(data)
    n_col = div(length(pred), n_row)
    return transpose(reshape(pred, (n_col, n_row)))
end


function save_model(bst::Booster, fname::String)
    XGBoosterSaveModel(bst.handle, fname)
    return nothing
end


# function save_rabit_checkpoint()


function save_raw(bst::Booster)
    return XGBoosterGetModelRaw(bst.handle)
end


function set_attr(bst::Booster;
                  kwargs...)
    for (attr, value) in kwargs
        XGBoosterSetAttr(bst.handle, string(attr), string(value))
    end
    return nothing
end


function set_param{T<:Any}(bst::Booster, params::Dict{String,T})
    for (param, value) in params
        XGBoosterSetParam(bst.handle, param, string(value))
    end
    return nothing
end


function set_param(bst::Booster, param::String;
                   value::Any = nothing)
    XGBoosterSetParam(bst.handle, param, string(value))
    return nothing
end


function update(bst::Booster, dtrain::DMatrix, iteration::Integer;
                fobj::Union{Function,Void} = nothing)
    if typeof(fobj) == Function
        pred = predict(bst, dtrain)
        grad, hess = fobj(pred, dtrain)
        @assert size(grad) == size(hess)
        XGBoosterBoostOneIter(bst.handle, dtrain.handle, grad, hess, length(hess))
    else
        XGBoosterUpdateOneIter(bst.handle, iteration, dtrain.handle)
    end
    return nothing
end
