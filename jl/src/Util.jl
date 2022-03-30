module Util

using CSV, DataFrames, Discretizers, Distances, Interpolations, MLDataUtils, Printf, PyCall, Random, ScikitLearn
using CherenkovDeconvolution, MetaConfigurations
import LinearAlgebra

DISTANCE_EPSILON = 1e-9 # min value of pdfs assumed for distance computations

"""
    SkObject(class_name, args...; kwargs...)

This compilation-ready version of `ScikitLearn.@sk_import` calls the constructor of the
fully-qualified `class_name` with the given `args` and `kwargs`.
"""
function SkObject(class_name::AbstractString, args...; kwargs...)
    # @info "The import uses scikit-learn $(pyimport(split(class_name, ".")[1]).__version__)"
    Constructor = getproperty(
        pyimport(join(split(class_name, ".")[1:end-1], ".")), # package name
        Symbol(split(class_name, ".")[end]) # un-qualified class name
    )
    return Constructor(args...; kwargs...)
end

# in-place substitutes of ScikitLearn.@sk_import
DecisionTreeClassifier(args...; kwargs...) = SkObject("sklearn.tree.DecisionTreeClassifier", args...; kwargs...)
LogisticRegression(args...; kwargs...) = SkObject("sklearn.linear_model.LogisticRegression", args...; kwargs...)
Pipeline(args...; kwargs...) = SkObject("sklearn.pipeline.Pipeline", args...; kwargs...)
StandardScaler(args...; kwargs...) = SkObject("sklearn.preprocessing.StandardScaler", args...; kwargs...)
CalibratedClassifierCV(args...; kwargs...) = SkObject("sklearn.calibration.CalibratedClassifierCV", args...; kwargs...)
KFold(args...; kwargs...) = SkObject("sklearn.model_selection.KFold", args...; kwargs...)
label_binarize(args...; kwargs...) = SkObject("sklearn.preprocessing.label_binarize", args...; kwargs...)
IsotonicRegression(args...; kwargs...) = SkObject("sklearn.isotonic.IsotonicRegression", args...; kwargs...)
CountVectorizer(args...; kwargs...) = SkObject("sklearn.feature_extraction.text.CountVectorizer", args...; kwargs...)

"""
    read_csv(path; nrows=nothing) = DataFrames.disallowmissing!(CSV.read(path; limit=nrows))

Read the CSV file, not allowing missing values. `nrows` specifies the number of rows to be read.
"""
read_csv(path::AbstractString; nrows::Union{Integer, Nothing}=nothing) =
    DataFrames.disallowmissing!(CSV.read(path, DataFrame; limit=nrows))

"""
    classifier_from_config(config)

Configure a classifier from a YAML file path or from a configuration Dict `config`.
"""
classifier_from_config(config::AbstractString) =
    classifier_from_config(parsefile(config; dicttype=Dict{Symbol,Any}))
function classifier_from_config(config::Dict{Symbol,Any})
    classname = config[:classifier]
    parameters = haskey(config, :parameters) ? config[:parameters] : Dict{Symbol,Any}()
    preprocessing = get(config, :preprocessing, "")
    calibration = Symbol(get(config, :calibration, "none"))
    return classifier(classname, preprocessing, calibration;
                      zip(Symbol.(keys(parameters)), values(parameters))...)
end

"""
    classifier(classname, preprocessing = "", calibration = :none; kwargs...)

Configure a classifier to be used in `train_and_predict_proba`. The `preprocessing`
can be empty (default), or the name of a scikit-learn transformer like `"StandardScaler"`.

The `calibration` can be `:none` (default), or `:isotonic`.

The keyword arguments configure the corresponding class (see official scikit-learn doc).
"""
function classifier(classname::AbstractString,
                    preprocessing::AbstractString = "",
                    calibration::Symbol = :none;
                    kwargs...)
    # instantiate classifier object
    Classifier = eval(Meta.parse(classname)) # constructor method
    classifier = Classifier(; kwargs...)

    # add calibration
    if calibration == :isotonic
        classifier = CalibratedClassifierCV(
            classifier,
            method=string(calibration),
            cv=KFold(n_splits=3) # do not stratify CV
        )
    elseif calibration != :none
        throw(ArgumentError("calibration has to be :none, or :isotonic"))
    end

    # add pre-processing
    if preprocessing != ""
        transformer = eval(Meta.parse(preprocessing))() # call the constructor method
        classifier  = Pipeline([ ("preprocessing", transformer), ("classifier", classifier) ])
    end
    return classifier
end

"""
    rnod(a, b)

Root Normalized Order-aware Divergence (RNOD) [sakai2021evaluating].
"""
rnod(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    sqrt(__od(a, b) / (length(a) - 1))

"""
    rsnod(a, b)

Root Symmetric Normalized Order-aware Divergence (RSNOD) [sakai2021evaluating].
"""
rsnod(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    sqrt((__od(a, b)/2 + __od(b, a)/2) / (length(a) - 1))

function __od(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number
    __check_distance_arguments(a, b)
    d = (a - b) .^ 2 # (p_j - p*_j)^2 for all j
    DW = i -> sum(abs(i - j) * d[j] for j in 1:length(b)) # Eq. 12 in [sakai2021evaluating]
    C_star = findall(b .> 0) # C*, the classes with non-zero probability
    return sum(DW, C_star) / length(C_star) # Eq. 13 in [sakai2021evaluating]
end

__check_distance_arguments(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    if length(a) != length(b)
        throw("length(a) = $(length(a)) != length(b) = $(length(b))")
    elseif !isapprox(sum(a), sum(b))
        throw("histograms have to have the same mass (difference is $(sum(a)-sum(b))")
    end

"""
    nmd(a, b) = mdpa(a, b) / (length(a) - 1)

Compute the Normalized Match Distance (NMD) [sakai2021evaluating], a variant of the Earth
Mover's Distance [rubner1998metric] which is normalized by the number of classes.
"""
nmd(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    mdpa(a, b) / (length(a) - 1)

"""
    mdpa(a, b)

Minimum Distance of Pair Assignments (MDPA) [cha2002measuring] for ordinal pdfs `a` and `b`.
The MDPA is a special case of the Earth Mover's Distance [rubner1998metric] that can be
computed efficiently.
"""
function mdpa(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number
    __check_distance_arguments(a, b)
    prefixsum = 0.0 # algorithm 1 in [cha2002measuring]
    distance  = 0.0
    for i in 1:length(a)
        prefixsum += a[i] - b[i]
        distance  += abs(prefixsum)
    end
    return distance / sum(a) # the normalization is a fix to the original MDPA
end

end # module
