module Data

using CSV, DataFrames, Distributions, Discretizers, HDF5, Random, StatsBase
using CherenkovDeconvolution.DeconvUtil: normalizepdf
using MetaConfigurations: parsefile
using ..Util

# import UCIData without precompiling because its precompilation is broken
__init__() = @eval using UCIData

DISCRETIZATION_Y = "discretization_y" # property in config files

"""
    LinearDiscretizer(configfile, key=""; kwargs...)

Configure a `LinearDiscretizer` object with the YAML `configfile`.

The YAML file should specify the properties `min`, `max`, and `num_bins`. If `key` is not
empty, these properties are looked for under the given `key`. All of these properties can be
substituted with the keyword arguments, e.g. with `min = 0.4`.

Example configuration file
--------------------------

    # myconf.yml
    mykey:
      min: 0.3
      max: 0.8
      num_bins: 3
"""
function Discretizers.LinearDiscretizer(
        configfile::String,
        key::String = DISCRETIZATION_Y;
        kwargs...)
    # load config
    c = parsefile(configfile)
    if length(key) > 0
        c = c[key]
    end
    
    # merge arguments into config
    for (k, v) in kwargs
        c[string(k)] = v
    end
    
    return LinearDiscretizer(range(c["min"], stop=c["max"], length=c["num_bins"]+1))
end

"""
    bins(ld)

Obtain the bin indices of the discretizer (or DataSet object) `ld`.
"""
bins(ld::LinearDiscretizer) = collect(1:length(bincenters(ld))) # DataSet impl is below
bins(cd::CategoricalDiscretizer) = collect(1:nlabels(cd))


# DataSet type implementations
abstract type DataSet end   # abstract supertype
include("data/fact.jl")     # FACT telescope data
include("data/ordinal.jl")  # ordinal data from UCI and OpenML

"""
    X_data(d)

Return the feature matrix `X` of observed data in the DataSet `d`.
"""
X_data(d::DataSet) = throw(ArgumentError("X_data is not implemented for $(typeof(d))"))

"""
    y_data(d)

Return the target value array `y` of observed data in the DataSet `d`.
"""
y_data(d::DataSet) = throw(ArgumentError("y_data is not implemented for $(typeof(d))"))

"""
    X_train(d)

Return the feature matrix `X` of training data in the DataSet `d`.
"""
X_train(d::DataSet) = throw(ArgumentError("X_train is not implemented for $(typeof(d))"))

"""
    y_train(d)

Return the target value array `y` of training data in the DataSet `d`.
"""
y_train(d::DataSet) = throw(ArgumentError("y_train is not implemented for $(typeof(d))"))

"""
    discretizer(d)

Return the target value discretizer for the DataSet `d`.
"""
discretizer(d::DataSet) = throw(ArgumentError("discretizer is not implemented for $(typeof(d))"))

bins(d::DataSet) = bins(discretizer(d))

"""
    dataset(id)

Return the DataSet object with the given `id`.
"""
dataset(id::AbstractString, args...; kwargs...) =
    if id == "fact"
        return Fact(args...; kwargs...)
    elseif id ∈ keys(parsefile("conf/data/uci.yml")) 
        return UciDataSet(id; kwargs...)
    elseif id ∈ keys(parsefile("conf/data/openml.yml"))
        return OpenmlDataSet(id; kwargs...)
    else
        throw(KeyError(id))
    end

struct ExhaustedClassException <: Exception
    label::Int64 # the label that is exhausted
    desired::Int64
    available::Int64
end
Base.showerror(io::IO, e::ExhaustedClassException) = print(io,
    "ExhaustedClassException: Cannot sample $(e.desired) instances of class $(e.label)",
    " (only $(e.available) available)")

"""
    subsample_indices([rng,] y, p, N)

Subsample `N` indices of labels `y` with prevalences `p`, using the random
number generator `rng`.

    subsample_indices([rng,] y, p_trn, N_trn, p_tst, N_tst)

Subsample indices of a training test split of the labels `y`. The training
and test sets have the class prevalences `p_trn` and `p_tst`, respectively,
and `N_trn` and `N_tst` samples. The two sets are distinct.

If the `rng` argument is omitted, both versions of `subsample_indices` use
the global random number generator `Random.GLOBAL_RNG`.
"""
function subsample_indices(
        rng::AbstractRNG,
        y::AbstractVector{Int},
        p::AbstractVector{R},
        N::Int;
        n_classes::Int=length(unique(y)),
        allow_duplicates::Bool=true,
        min_samples_per_class::Int=3 # ensure that calibration works
        ) where {R<:Real}
    if length(p) != n_classes
        throw(ArgumentError("length(p) != n_classes"))
    end
    to_take = max.(round.(Int, N * p), min_samples_per_class)
    while N != sum(to_take)
        N_remaining = N - sum(to_take)
        if N_remaining > 0 # are additional draws needed?
            to_take[StatsBase.sample(rng, 1:n_classes, Weights(max.(p, 1/N)), N_remaining)] .+= 1
        elseif N_remaining < 0 # are less draws needed?
            c = findall(to_take .> min_samples_per_class)
            to_take[StatsBase.sample(rng, c, Weights(max.(p[c], 1/N)), -N_remaining)] .-= 1
        end
    end # rarely needs more than one iteration
    i = randperm(rng, length(y)) # random order after shuffling
    j = vcat(map(1:n_classes) do c
        i_c = (1:length(y))[y[i].==c]
        if to_take[c] <= length(i_c)
            i_c[1:to_take[c]] # the return value of this map operation
        elseif allow_duplicates
            @debug "Have to repeat $(ceil(Int, to_take[c] / length(i_c))) times"
            i_c = repeat(i_c, ceil(Int, to_take[c] / length(i_c)))
            i_c[1:to_take[c]] # take from a repetition
        else
            throw(ExhaustedClassException(c, to_take[c], length(i_c)))
        end
    end...) # indices of the shuffled sub-sample
    return i[j]
end

subsample_indices(
        y::AbstractVector{Int},
        p::AbstractVector{R},
        N::Int;
        kwargs...
        ) where {R<:Real} =
    subsample_indices(Random.GLOBAL_RNG, y, p, N; kwargs...)

# integrated training test split
function subsample_indices(
        rng::AbstractRNG,
        y::AbstractVector{Int},
        p_trn::AbstractVector{R},
        N_trn::Int,
        p_tst::AbstractVector{R},
        N_tst::Int;
        min_samples_per_class::Tuple{Int,Int}=(3, 3),
        kwargs...
        ) where {R<:Real}
    i_tst = subsample_indices(
        rng, y, p_tst, N_tst; min_samples_per_class=min_samples_per_class[1], kwargs...
    )
    i_rem = setdiff(1:length(y), i_tst) # remaining indices
    i_trn = i_rem[subsample_indices(
        rng, y[i_rem], p_trn, N_trn; min_samples_per_class=min_samples_per_class[2], kwargs...
    )]
    return i_trn, i_tst
end

subsample_indices(
        y::AbstractVector{Int},
        p_trn::AbstractVector{R},
        N_trn::Int,
        p_tst::AbstractVector{R},
        N_tst::Int;
        kwargs...
        ) where {R<:Real} =
    subsample_indices(Random.GLOBAL_RNG, y, p_trn, N_trn, p_tst, N_tst; kwargs...)

end # module
