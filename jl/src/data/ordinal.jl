using SparseArrays
using PyCallUtils

"""
    struct OrdinalDataSet <: DataSet

The type of ordinal datasets, as initialized through `OpenmlDataSet(name)` or
`UciDataSet(name)`.

**See also:** `X_data`, `y_data`, `discretizer`
"""
struct OrdinalDataSet <: DataSet
    X_data::Matrix{Float32} # sklearn handles Float32
    y_data::Vector{Float32}
    discretizer::AbstractDiscretizer
end

# implementation of the interface
X_data(d::OrdinalDataSet) = d.X_data
y_data(d::OrdinalDataSet) = d.y_data
discretizer(d::OrdinalDataSet) = d.discretizer

""""
    OpenmlDataSet(name; config, min, max, num_bins)

Retrieve an OpenML `OrdinalDataSet` by its `name`.

The keyword arguments `min`, `max`, and `num_bins` define the discretizer of the
resulting data set. The default values of these parameters are specified in the
`config` dictionary, as read from `conf/data/openml.yml`.
"""
function OpenmlDataSet(
        name::String;
        config::Dict{String,Any}=parsefile("conf/data/openml.yml")[name],
        min::Real=NaN,
        max::Real=NaN,
        num_bins::Int=0,
        readdata::Bool=true)
    if readdata
        bunch = Util.SkObject(
            "sklearn.datasets.fetch_openml";
            data_id=config["id"],
            as_frame=true
        ) # a Dict of pandas objects
        df = DataFrame(; map(
            c -> Symbol(c) => bunch["data"][c].to_numpy(),
            bunch["data"].columns
        )...)
        min = isnan(min) ? config[DISCRETIZATION_Y]["min"] : min
        max = isnan(max) ? config[DISCRETIZATION_Y]["max"] : max
        y_c = [ bunch["target"].values... ] # continuous values
        i = (y_c .>= min) .& (y_c .<= max) # selection
        rename!(df, names(df) .=> map(1:length(names(df))) do j
            j in get(config, "categorical_columns", String[]) ? "C_$j" : "N_$j"
        end) # column names C_n and N_n define the column type, just like in the UciDataSet
        X = _extract_features(df)
        y = _extract_labels(y_c, config)
    else
        X = zeros(Float32, 0, 0)
        y = zeros(Float32, 0)
    end
    d = _configure_discretizer(config, min, max, num_bins)
    return OrdinalDataSet(X, y, d)
end

""""
    UciDataSet(name; config, min, max, num_bins)

Retrieve a UCI `OrdinalDataSet` by its `name`.

The keyword arguments `min`, `max`, and `num_bins` define the discretizer of the
resulting data set. The default values of these parameters are specified in the
`config` dictionary, as read from `conf/data/uci.yml`.
"""
function UciDataSet(
        name::String;
        config::Dict{String,Any}=parsefile("conf/data/uci.yml")[name],
        min::Real=NaN,
        max::Real=NaN,
        num_bins::Int=0,
        readdata::Bool=true)
    if readdata
        df = UCIData.dataset(name)
        dropmissing!(df, disallowmissing=true)
        min = isnan(min) ? config[DISCRETIZATION_Y]["min"] : min
        max = isnan(max) ? config[DISCRETIZATION_Y]["max"] : max
        i = (df[!, :target] .>= min) .& (df[!, :target] .<= max) # selection
        X = _extract_features(df[i, :])
        y = _extract_labels(df[i, :target], config)
    else
        X = zeros(Float32, 0, 0)
        y = zeros(Float32, 0)
    end
    d = _configure_discretizer(config, min, max, num_bins)
    return OrdinalDataSet(X, y, d)
end

# extract ordinal or regression labels to be binned
function _extract_labels(targets::AbstractVector, config::Dict{String,Any})
    if "labels" in keys(config) # ordinal labels
        d = CategoricalDiscretizer(Dict(zip(config["labels"], 1:length(config["labels"]))))
        return Vector{Int32}(encode(d, string.(targets)))
    elseif eltype(targets) <: Real # regression labels of any Real type
        return Float32.(targets) # type cast
    else # regression labels of type String
        return parse.(Float32, targets) # parsing != type cast
    end
end

# configure the discretizer
function _configure_discretizer(config::Dict{String,Any}, min::Real, max::Real, num_bins::Int)
    if haskey(config, DISCRETIZATION_Y) || !isnan(min) || !isnan(max) || num_bins > 0
        if !haskey(config, DISCRETIZATION_Y) && (isnan(min) || isnan(max) || num_bins < 1)
            throw(ArgumentError("Need to specify min, max, and num_bins"))
        end
        min = isnan(min) ? config[DISCRETIZATION_Y]["min"] : min
        max = isnan(max) ? config[DISCRETIZATION_Y]["max"] : max
        num_bins = num_bins < 1 ? config[DISCRETIZATION_Y]["num_bins"] : num_bins
        return LinearDiscretizer(range(min, stop=max, length=num_bins+1))
    else
        bins_y = 1:ceil(Int, maximum(y_data))
        return CategoricalDiscretizer(Dict(zip(bins_y, bins_y))) # a dummy mapping
    end
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
