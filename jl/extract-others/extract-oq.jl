#
# julia extract-others.jl
#
using CSV, DataFrames, Discretizers, PyCall, UCIData
using MetaConfigurations: parsefile

py"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""

function _read_OpenML(name::String)
    config = parsefile("openml.yml")[name]
    bunch = pyimport("sklearn.datasets").fetch_openml(
        data_id = config["id"],
        as_frame = true
    ) # load a Dict of pandas objects
    df = DataFrame(; map(
        c -> Symbol(c) => bunch["data"][c].to_numpy(),
        bunch["data"].columns
    )...)
    y_c = [ bunch["target"].values... ] # continuous values
    rename!(df, names(df) .=> map(j -> "N_$j", 1:length(names(df))))
    X = _extract_features(df)
    y = _extract_labels(y_c)
    d = _configure_discretizer(config)
    return _prepare_DataFrame(X, y, d)
end

function _read_UCI(name::String)
    config = parsefile("uci.yml")[name]
    df = UCIData.dataset(name)
    dropmissing!(df, disallowmissing=true)
    min = config["discretization_y"]["min"]
    max = config["discretization_y"]["max"]
    i = (df[!, :target] .>= min) .& (df[!, :target] .<= max) # selection
    X = _extract_features(df[i, :])
    y = _extract_labels(df[i, :target])
    d = _configure_discretizer(config)
    return _prepare_DataFrame(X, y, d)
end

function _prepare_DataFrame(X::AbstractMatrix, y::AbstractVector, d::LinearDiscretizer)
    df = DataFrame(class_label = encode(d, y)) # regression target -> ordinal class
    for (i, c) in enumerate(eachcol(X))
        df[!, "feature_$i"] = convert.(Float32, c)
    end
    return df
end

# extract a matrix of numerical features
function _extract_features(df::DataFrame)
    C = filter(c -> startswith(string(c), "C"), names(df)) # categorical features
    N = filter(n -> startswith(string(n), "N"), names(df)) # numerical features
    X = Array{Float32}(undef, size(df, 1), length(C) + length(N))
    for (i, c) in enumerate(C)
        X[:, i] = encode(CategoricalDiscretizer(df[!, c]), df[!, c])
    end
    for (i, n) in enumerate(N)
        X[:, i+length(C)] = df[!, n]
    end
    return X
end

# extract regression labels to be binned
function _extract_labels(targets::AbstractVector)
    if eltype(targets) <: Real # regression labels of any Real type
        return Float32.(targets) # type cast
    else # regression labels of type String
        return parse.(Float32, targets) # parsing != type cast
    end
end

# configure the discretizer
_configure_discretizer(config::Dict{String,Any}) =
    LinearDiscretizer(range(
        config["discretization_y"]["min"],
        stop = config["discretization_y"]["max"],
        length = config["discretization_y"]["num_bins"]+1
    ))

# extract OpenML data
for name in ["Yolanda", "fried"]
    @info "Loading $name"
    df = _read_OpenML(name)
    @info "Writing prepared data to $name.csv"
    CSV.write("$name.csv", df)
end

# extract UCI data
for name in ["blog-feedback", "online-news-popularity"]
    @info "Loading $name"
    df = _read_UCI(name)
    @info "Writing prepared data to $name.csv"
    CSV.write("$name.csv", df)
end
