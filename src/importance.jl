
# available feature type options:
#   weight, gain, cover, total_gain, total_cover

# they seem to only ever name features this way, regardless of whether you set them
_parse_out_feature_name(str::AbstractString) = parse(Int, lstrip(str, 'f'))+1

#TODO: still have to figure out their refusal to name features despite nominally supporting it

function importance(b::Booster; type::AbstractString="gain")
    cfg = JSON3.write(Dict("importance_type"=>type))

    olen_fnames = Ref{Lib.bst_ulong}()
    o_fnames = Ref{Ptr{Ptr{Cchar}}}()

    olen_shape = Ref{Lib.bst_ulong}()
    o_shape = Ref{Ptr{Lib.bst_ulong}}()

    o = Ref{Ptr{Cfloat}}()

    xgbcall(XGBoosterFeatureScore, b.handle, cfg, olen_fnames, o_fnames, olen_shape, o_shape, o)

    names = unsafe_wrap(Array, o_fnames[], olen_fnames[])
    names = map(unsafe_string, names)

    dims = unsafe_wrap(Array, o_shape[], olen_shape[])
    dims = reduce(tuple, dims)

    fim = unsafe_wrap(Array, o[], dims)

    if fim isa AbstractVector  # ensure indexing below works regardless of shape
        fim = reshape(fim, (length(fim),1))
    end

    fim = map(j -> fim[j,:], 1:size(fim,1))

    p = sortperm(fim, by=sum, rev=true)

    OrderedDict(_parse_out_feature_name.(names[p]) .=> fim[p])
end
