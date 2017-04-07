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


"""
    get_base_margin(dmat)

Return the base margin of the DMatrix as Vector{Cfloat}.

# Arguments
* `dmat::DMatrix`: the DMatrix.
"""
function get_base_margin(dmat::DMatrix)
    return XGDMatrixGetFloatInfo(dmat.handle, "base_margin")
end


"""
    get_float_info(dmat, field)

Return float property of the DMatrix as Vector{Cfloat}.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `field::String`: the field name of the property.
"""
function get_float_info(dmat::DMatrix, field::String)
    return XGDMatrixGetFloatInfo(dmat.handle, field)
end


"""
    get_label(dmat)

Return the label of the DMatrix as Vector{Cfloat}.

# Arguments
* `dmat::DMatrix`: the DMatrix.
"""
function get_label(dmat::DMatrix)
    return XGDMatrixGetFloatInfo(dmat.handle, "label")
end


"""
    get_uint_info(dmat, field)

Return unsigned integer property from the DMatrix as Vector{Cuint}.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `field::String`: the field name of the information.
"""
function get_uint_info(dmat::DMatrix, field::String)
    return XGDMatrixGetUIntInfo(dmat.handle, field)
end


"""
    get_weight(dmat)

Return the weight of the DMatrix as Vector{Cfloat}.

# Arguments
* `dmat::DMatrix`: the DMatrix.
"""
function get_weight(dmat::DMatrix)
    return XGDMatrixGetFloatInfo(dmat.handle, "weight")
end


"""
    num_col(dmat)

Return the number of columns (features) in the DMatrix as Int.

# Arguments
* `dmat::DMatrix`: the DMatrix.
"""
function num_col(dmat::DMatrix)
    return Int(XGDMatrixNumCol(dmat.handle))
end


"""
    num_row(dmat)

Return the number of rows in the DMatrix as Int.

# Arguments
* `dmat::DMatrix`: the DMatrix.
"""
function num_row(dmat::DMatrix)
    return Int(XGDMatrixNumRow(dmat.handle))
end


"""
    save_binary(dmat, fname; [silent = true])

Save DMatrix to an XGBoost buffer.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `fname::String`: name of the output buffer file.
* `silent::Bool`: if set, the output is suppressed.
"""
function save_binary(dmat::DMatrix, fname::String;
                     silent::Bool = true)
    XGDMatrixSaveBinary(dmat.handle, fname, silent)
    return nothing
end


"""
    set_base_margin(dmat, margin)

Set base margin of booster to start from.

This can be used to specify a prediction value of existing model to be base_margin. However,
remember margin is needed, instead of transformed prediction e.g. for logistic regression: need to
put in value before logistic transformation.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `margin::Vector{T<:Real}`: prediction margin of each datapoint.
"""
function set_base_margin{T<:Real}(dmat::DMatrix, margin::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "base_margin", margin, length(margin))
    return nothing
end


"""
    set_float_info(dmat, field, data)

Set float type property into the DMatrix.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `field::String`: the field name of the information.
* `data::Vector{T<:Real}`: the array of data to be set.
"""
function set_float_info{T<:Real}(dmat::DMatrix, field::String, data::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, field, data, length(data))
    return nothing
end


"""
    set_group(dmat, group)

Set group size of DMatrix (used for ranking).

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `group::Vector{T<:Integer}`: group size of each group.
"""
function set_group{T<:Integer}(dmat::DMatrix, group::Vector{T})
    XGDMatrixSetGroup(dmat.handle, group, length(group))
    return nothing
end


"""
    set_label(dmat, label)

Set label of DMatrix.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `label::Vector{T<:Real}`: the label information to be set into DMatrix.
"""
function set_label{T<:Real}(dmat::DMatrix, label::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "label", label, length(label))
    return nothing
end


"""
    set_uint_info(dmat, field, data)

Set uint type property into the DMatrix.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `field::String`: the field name of the information.
* `data::Vector{T<:Integer}`: the array of data to be set.
"""
function set_uint_info{T<:Integer}(dmat::DMatrix, field::String, data::Vector{T})
    XGDMatrixSetUIntInfo(dmat.handle, field, data, length(data))
    return nothing
end


"""
    set_weight(dmat, weight)

Set weight of each instance.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `weight::Vector{T<:Real}`: weight for each data point.
"""
function set_weight{T<:Real}(dmat::DMatrix, weight::Vector{T})
    XGDMatrixSetFloatInfo(dmat.handle, "weight", weight, length(weight))
    return nothing
end


"""
    slice(dmat, rindex)

Slice the DMatrix and return a new DMatrix that only contains rindex.

# Arguments
* `dmat::DMatrix`: the DMatrix.
* `rindex::Vector{T<:Integer}`: list of indices to be selected.
"""
function slice{T<:Integer}(dmat::DMatrix, rindex::Vector{T})
    handle = XGDMatrixSliceDMatrix(dmat.handle, rindex - 1, length(rindex))
    return DMatrix(handle)
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


"""
    attr(bst, attr)

Return attribute from the Booster as a String.

# Arguments
* `bst::Booster`: the Booster.
* `attr:String`: the key to get attribute from.
"""
function attr(bst::Booster, attr::String)
    return XGBoosterGetAttr(bst.handle, attr)
end


"""
    attributes(bst)

Return all attributes stored in the Booster as a Dict{String,String}.

# Arguments
* `bst::Booster`: the Booster.
"""
function attributes(bst::Booster)
    attr_names = XGBoosterGetAttrNames(bst)
    result = Dict{String,String}()
    for attr_name in attr_names
        result[attr_name] = XGBoosterGetAttr(bst.handle, attr_name)
    end
    return result
end


"""
    boost(bst, dtrain, grad, hess)

Boost the Booster for one iteration, with customized gradient statistics.

# Arguments
* `bst::Booster`: the Booster.
* `dtrain::DMatrix`: the training DMatrix.
* `grad::Vector{Float32}`: the first order of the gradient.
* `hess::Vector{Float32}`: the second order of the gradient.
"""
function boost(bst::Booster, dtrain::DMatrix, grad::Vector{Float32}, hess::Vector{Float32})
    @assert size(grad) == size(hess)
    XGBoosterBoostOneIter(bst.handle, dtrain.handle, grad, hess, length(hess))
    return nothing
end


"""
    copy(bst)

Return a copy of the Booster.

# Arguments
* `bst::Booster`: the Booster.
"""
function copy(bst::Booster)
    return Booster(model_file = save_raw(bst))
end


"""
    dump_model(bst, fout; [fmap = "", with_stats = false])

Dump the model in the Booster into a text file.

# Arguments
* `bst::Booster`: the Booster.
* `fout::String`: output file name.
* `fmap::String`: name of the file containing feature map names.
* `with_stats::Bool`: controls whether the split statistics are output.
"""
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


"""
    eval(bst, data; [name = "eval", iteration = 0])

Return an evaluation of the model on the data as String.

# Arguments
* `bst::Booster`: the Booster.
* `data::DMatrix`: the DMatrix storing the input.
* `name::String`: the name of the dataset.
* `iteration::Int`: the current iteration number.
"""
function eval(bst::Booster, data::DMatrix;
              name::String = "eval", iteration::Int = 0)
    return eval_set(bst, [(data, name)], iteration)
end


"""
    eval_set(bst, evals, iteration; [feval = nothing])

Return multiple evaluations of the model as String.

# Arguments
* `bst::Booster`: the Booster.
* `evals::Vector{Tuple{DMatrix,String}}`: list of items to be evaluated.
* `iteration::Integer`: the current iteration number.
* `feval::Union{Function,Void}`: custom evaluation function or `nothing`.
"""
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


# TODO: Make sure this returns a Vector{String}.
"""
    get_dump(bst; [fmap = "", with_stats = false])

Returns a dump of the model as a Vector{String}.

# Arguments
* `bst::Booster`: the Booster.
* `fmap::String`: name of the file containing feature map names.
* `with_stats::Bool`: controls whether the split statistics are output.
"""
function get_dump(bst::Booster;
                  fmap = "", with_stats = false)
    raw_dump = XGBoosterDumpModel(bst.handle, fmap, with_stats)
    model = [unsafe_string(ptr) for ptr in raw_dump]
    return return model
end


# function get_fscore(bst::Booster; fmap = "")
# function get_score(bst::Booster; fmap = "", importance_type = "weight")
# function get_split_value_histogram(bst::Booster, feature::String; fmap = "", bins = nothing)


"""
    load_model(fname)

Return a Booster with the model loaded from a file.

# Arguments
* `fname::String`: input file name.
"""
function load_model(fname::String)
    bst = Booster()
    XGBoosterLoadModel(bst.handle, fname)
    return bst
end


"""
    load_model(fname)

Return a Booster with the model loaded from an in-memory buffer representation.

# Arguments
* `fname::Vector{UInt8}`: Input memory buffer.
"""
function load_model(fname::Vector{UInt8})
    bst = Booster()
    len = length(fname)
    XGBoosterLoadModelFromBuffer(bst.handle, fname, len)
    return bst
end


# function load_rabit_checkpoint()


"""
    predict(bst, data; [output_margin = false, ntree_limit = 0])

Return preditions for the data as Vector{Cfloat}.

# Arguments
* `bst::Booster`: the Booster.
* `data::DMatrix`: the DMatrix storing the input.
* `output_margin::Bool`: whether to output the raw untransformed margin value.
* `ntree_limit::Integer`: limit number of trees in the prediction; defaults to 0 (use all trees).
"""
function predict(bst::Booster, data::DMatrix;
                 output_margin::Bool = false, ntree_limit::Integer = 0)
    option_mask = 0x00
    if output_margin
        option_mask |= 0x01
    end

    return XGBoosterPredict(bst.handle, data.handle, option_mask, ntree_limit)
end


"""
    predict_leaf(bst, data; [ntree_limit = 0])

Return preditions for the leaf indices of the data as Matrix{Cfloat}.

# Arguments
* `bst::Booster`: the Booster.
* `data::DMatrix`: the DMatrix storing the input.
* `ntree_limit::Integer`: limit number of trees in the prediction; defaults to 0 (use all trees).
"""
function predict_leaf(bst::Booster, data::DMatrix;
                      ntree_limit::Integer = 0)
    option_mask = 0x02
    pred = XGBoosterPredict(bst.handle, data.handle, option_mask, ntree_limit)
    n_row = num_row(data)
    n_col = div(length(pred), n_row)
    return transpose(reshape(pred, (n_col, n_row)))
end


"""
    save_model(bst, fname)

Save the model to a file.

# Arguments
* `bst::Booster`: the Booster.
* `fname::String`: output file name.
"""
function save_model(bst::Booster, fname::String)
    XGBoosterSaveModel(bst.handle, fname)
    return nothing
end


# function save_rabit_checkpoint()


"""
    save_raw(bst)

Return an in-memory buffer representation of the model as Vector{UInt8}.

# Arguments
* `bst::Booster`: the Booster.
"""
function save_raw(bst::Booster)
    return XGBoosterGetModelRaw(bst.handle)
end


# TODO: Make sure that deleting attributes works correctly.
"""
    set_attr(bst; [kwargs...])

Set the attribute of the Booster.

# Arguments
* `bst::Booster`: the Booster.
* `kwargs...`: the attributes to set. Setting a value to `nothing` deletes an attribute.
"""
function set_attr(bst::Booster;
                  kwargs...)
    for (attr, value) in kwargs
        XGBoosterSetAttr(bst.handle, string(attr), string(value))
    end
    return nothing
end


"""
    set_param(bst, params)

Set parameters into the Booster.

# Arguments
* `bst::Booster`: the Booster.
* `params::Dict{String,T<:Any}`: dictionary of (key, value) pairs.
"""
function set_param{T<:Any}(bst::Booster, params::Dict{String,T})
    for (param, value) in params
        XGBoosterSetParam(bst.handle, param, string(value))
    end
    return nothing
end


"""
    set_param(bst, param; [value = nothing])

Set a parameter in the Booster.

# Arguments
* `bst::Booster`: the Booster.
* `param::String`: key of the parameter to set.
* `value::Any`: the value to set the parameter to.
"""
function set_param(bst::Booster, param::String;
                   value::Any = nothing)
    XGBoosterSetParam(bst.handle, param, string(value))
    return nothing
end


"""
    update(bst, dtrain, iteration; [fobj = nothing])

Update the Booster for one iteration, with objective function calculated internally.

# Arguments
* `bst::Booster`: the Booster.
* `dtrain::DMatrix`: training data.
* `iteration::Integer`: current iteration number.
* `fobj::Union{Function,Void}`: customized objective function or `nothing`.
"""
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
