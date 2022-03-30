module Util

using CSV, DataFrames, Discretizers, Distances, Interpolations, MLDataUtils, Printf, PyCall, Random, ScikitLearn
using CherenkovDeconvolution, ComfyCommons, MetaConfigurations
import LinearAlgebra

DISTANCE_EPSILON = 1e-9 # min value of pdfs assumed for distance computations

# need to implement the dummy classifier in Python...
__init__() = py"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class PyDummyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, X_trn=[], y_trn=[]):
        self.X_trn = X_trn
        self.y_trn = y_trn
    def fit(self, X_i, y): # X_i is only the index of a former training sample
        y_trn = np.array(self.y_trn, dtype=int)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.binary_ = True
            self.negative_classes_ = np.unique(y_trn[X_i[y==self.classes_[0]]])
            self.positive_classes_ = np.unique(y_trn[X_i[y==self.classes_[1]]])
        elif len(self.classes_) < len(np.unique(y_trn)): # some other subsampling has happened
            raise ValueError("Non-binary subsampling is not supported")
        else:
            self.binary_ = False
        return self
    def __binarize(self, X):
        if not isinstance(X, np.ndarray):
            X = np.concatenate(X) # needed for ARC
            if len(X.shape) == 1:
                X = X.reshape(1, -1) # a single sample is predicted
        if X.dtype in [np.int64, np.int32]: # training set prediction in OQT/ARC
            X = self.X_trn[X.flatten()]
        if not self.binary_:
            return X
        return np.hstack((
            X[:, self.negative_classes_].sum(axis=1, keepdims=True),
            X[:, self.positive_classes_].sum(axis=1, keepdims=True)
        ))
    def predict_proba(self, X):
        return self.__binarize(X)
    def predict(self, X):
        return self.classes_[self.__binarize(X).argmax(axis=1)]
"""
DummyClassifier(; X_trn=zeros(0, 0), y_trn=Int[]) = py"PyDummyClassifier"(X_trn, y_trn)

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
GaussianNB(args...; kwargs...) = SkObject("sklearn.naive_bayes.GaussianNB", args...; kwargs...)
RandomForestClassifier(args...; kwargs...) = SkObject("sklearn.ensemble.RandomForestClassifier", args...; kwargs...)
SVC(args...; kwargs...) = SkObject("sklearn.svm.SVC", args...; kwargs...)
KNeighborsClassifier(args...; kwargs...) = SkObject("sklearn.neighbors.KNeighborsClassifier", args...; kwargs...)
DecisionTreeClassifier(args...; kwargs...) = SkObject("sklearn.tree.DecisionTreeClassifier", args...; kwargs...)
LogisticRegression(args...; kwargs...) = SkObject("sklearn.linear_model.LogisticRegression", args...; kwargs...)
LinearSVC(args...; kwargs...) = SkObject("sklearn.svm.LinearSVC", args...; kwargs...)
Pipeline(args...; kwargs...) = SkObject("sklearn.pipeline.Pipeline", args...; kwargs...)
StandardScaler(args...; kwargs...) = SkObject("sklearn.preprocessing.StandardScaler", args...; kwargs...)
CalibratedClassifierCV(args...; kwargs...) = SkObject("sklearn.calibration.CalibratedClassifierCV", args...; kwargs...)
KFold(args...; kwargs...) = SkObject("sklearn.model_selection.KFold", args...; kwargs...)
label_binarize(args...; kwargs...) = SkObject("sklearn.preprocessing.label_binarize", args...; kwargs...)
IsotonicRegression(args...; kwargs...) = SkObject("sklearn.isotonic.IsotonicRegression", args...; kwargs...)
_SigmoidCalibration(args...; kwargs...) = SkObject("sklearn.calibration._SigmoidCalibration", args...; kwargs...)
CountVectorizer(args...; kwargs...) = SkObject("sklearn.feature_extraction.text.CountVectorizer", args...; kwargs...)
KMeans(args...; kwargs...) = SkObject("sklearn.cluster.KMeans", args...; kwargs...)
Nystroem(args...; kwargs...) = SkObject("sklearn.kernel_approximation.Nystroem", args...; kwargs...)

"""
    read_csv(path; nrows=nothing) = DataFrames.disallowmissing!(CSV.read(path; limit=nrows))

Read the CSV file, not allowing missing values. `nrows` specifies the number of rows to be read.
"""
read_csv(path::AbstractString; nrows::Union{Integer, Nothing}=nothing) =
    DataFrames.disallowmissing!(CSV.read(path, DataFrame; limit=nrows))


_DOC_STD = """
    std_maxl(f_est, R, g; density=true)
    std_lsq_poisson(R, g; density=true)
    std_lsq_multinomial(R, g; density=true)
    
    std_ibu_poisson(f_est, R, g; density=true)
    std_ibu_multinomial(f_est, R, g; density=true)
    
    std_f_poisson(f_est;    density=true)  where eltype(f_est) <: Integer
    std_f_poisson(f_est, N; density=true)  where eltype(f_est) <: Real

Estimate the standard deviation of each component in a deconvolution result `f_est`.

`std_maxl`, `std_lsq_poisson` and `std_lsq_multinomial` estimate the standard deviation from
the Hessian of their respective objective function, which is a likelihood function for
`std_maxl` and a least-squares objective for `std_lsq_poisson` and `std_lsq_multinomial`.
The first two of these methods assume a Poisson distribution in each of the *observed* bins
in `g`. The third one, `std_lsq_multinomial`, however, assumes a common multinomial
distribution of the observed bins. Least-squares estimates are independent of `f_est`
because the solution is unique in a least-squares fit.

A different approach is taken by `std_ibu_poisson` and `std_ibu_multinomial`, which use
Bayes' theorem to obtain a \"deconvolution matrix\" and then perform a standard uncertainty
propagation. Like above, `std_ibu_poisson` assumes Poisson-distributed *observed* bins and
`std_ibu_multinomial` assumes a common multinomial distribution.

Yet another assumption is made by `std_f_poisson`, which computes the standard deviation of
a Poisson distribution in each *result* bin `i`. The rate of this distribution is `f_est[i]`
if `f_est` consists of `Integer` elements and `f_est[i] * N` if `f_est` consists of `Real`
elements.

The keyword argument `density` specifies if the returned standard deviation is normalized to
the deviation from a density function. If not, the deviation is to be understood with
respect to absolute counts in each bin.
"""
@doc _DOC_STD std_maxl
@doc _DOC_STD std_lsq_poisson
@doc _DOC_STD std_lsq_multinomial
@doc _DOC_STD std_ibu_poisson
@doc _DOC_STD std_ibu_multinomial
@doc _DOC_STD std_f_poisson

function std_maxl( f_est :: AbstractVector{Tf},
                   R     :: Matrix{TR},
                   g     :: AbstractVector{Tg};
                   density::Bool=true ) where {Tf<:Number, TR<:Number, Tg<:Integer}
    N       = sum(g) # number of events
    f_abs   = convert.(Int64, round.(DeconvUtil.normalizepdf(f_est) .* N))
    return _std_H(CherenkovDeconvolution.Methods._maxl_H(R, g)(f_abs), density, N)
end

std_lsq_poisson( R :: Matrix{TR},
                 g :: AbstractVector{Tg};
                 density::Bool=true ) where {TR<:Number, Tg<:Integer} =
    _std_H(R'*LinearAlgebra.pinv(DeconvUtil.cov_Poisson(g))*R, density, sum(g)) # equivalent to using the Hessian from _lsq_H

std_lsq_multinomial( R :: Matrix{TR},
                     g :: AbstractVector{Tg};
                     density::Bool=true ) where {TR<:Number, Tg<:Integer} =
    _std_H(R'*LinearAlgebra.pinv(DeconvUtil.cov_multinomial(g))*R, density, sum(g))

# estimate the standard deviation in each bin from a Hessian matrix H
function _std_H(H::Matrix{Float64}, density::Bool, N::Integer)
    std_abs = sqrt.(abs.(LinearAlgebra.diag(LinearAlgebra.pinv(H))))  # deviation with respect to absolute counts
    return density ? std_abs./N : std_abs # normalize to a density
end # TODO what about summing up each column in H?  ->  sqrt.(sum(abs.(LinearAlgebra.pinv(H)), 2))

std_ibu_poisson( f_est :: AbstractVector{Tf},
                 R     :: Matrix{TR},
                 g     :: AbstractVector{Tg};
                 density::Bool=true ) where {Tf<:Number, TR<:Number, Tg<:Integer} =
    _std_ibu(DeconvUtil.normalizepdf(f_est), R, DeconvUtil.cov_Poisson(g), sum(g), density)

std_ibu_multinomial( f_est :: AbstractVector{Tf},
                     R     :: Matrix{TR},
                     g     :: AbstractVector{Tg};
                     density::Bool=true ) where {Tf<:Number, TR<:Number, Tg<:Integer} =
    _std_ibu(DeconvUtil.normalizepdf(f_est), R, DeconvUtil.cov_multinomial(g), sum(g), density)

# estimate the standard deviation in each bin using uncertainty propagation and Bayes' theorem
function _std_ibu( f_0     :: AbstractVector{Float64},
                   R       :: Matrix{TR},
                   cov_g   :: Matrix{Tc},
                   N       :: Integer,
                   density :: Bool ) where {TR<:Number, Tc<:Number}
    M   = CherenkovDeconvolution.Methods._ibu_reverse_transfer(R, f_0) # motivated by Bayes' theorem
    cov_abs = M * cov_g * M'              # cov of f_0, estimated by uncertainty propagation
    std_abs = sqrt.(abs.(LinearAlgebra.diag(cov_abs)))  # the diagonal elements of the cov matrix
    return density ? std_abs./N : std_abs # normalize to a density
end

std_f_poisson(f_est::AbstractVector{T}, N::Integer; density::Bool=true) where T<:Real =
    std_f_poisson(convert.(Int64, round.(DeconvUtil.normalizepdf(f_est).*N)), density=density)

function std_f_poisson(f_est::AbstractVector{T}; density::Bool=true) where T<:Integer
    std_abs = sqrt.(f_est) # Poisson deviation of absolute counts - no Hessian required
    return density ? std_abs./sum(f_est) : std_abs
end


"""
    interpolate_b_splines(x, y, grid_size::Int)

Interpolate the knots specified by their `x` and `y` coordinates on a grid of `grid_size`. Returns
the interpolated x and y coordinates.
"""
function interpolate_b_splines(x::Vector{T}, y::Vector{T}, grid_size::Int) where T<:Real
    if length(x) != length(y)
        throw(ArgumentError("length(x) is $(length(x)), but must be equal to length(y)=$(length(y))"))
    end
    # http://juliamath.github.io/Interpolations.jl/latest/control/#Parametric-splines-1
    t = range(0., stop=1., length=length(x))
    A = hcat(x,y)
    itp = Interpolations.scale(Interpolations.interpolate(A, (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t, 1:2)
    tfine = range(0., stop=1., length=grid_size)
    return [itp(t,1) for t in tfine], [itp(t,2) for t in tfine]
end


"""
    CalibratedRandomForestClassifier(method=:isotonic; kwargs...)

A `RandomForestClassifier` parametrized by `kwargs`, which is calibrated based on the
out-of-bag decision function applied to the training set. The calibration `method` is either
`:isotonic` or `:sigmoid`.
"""
mutable struct CalibratedRandomForestClassifier
    method::Symbol # calibration method (:isotonic or :sigmoid)
    rf::PyObject   # RandomForestClassifier
    cal::Vector{PyObject} # calibrators
end
function CalibratedRandomForestClassifier(method::Symbol=:isotonic; kwargs...)
    if !in(method, [:isotonic, :sigmoid])
        throw(ArgumentError("method must be :isotonic or :sigmoid, but is $method"))
    end
    return CalibratedRandomForestClassifier(
        method,
        RandomForestClassifier(; oob_score=true, filter(f -> f[1] .!= :oob_score, kwargs)...),
        PyObject[]
    )
end

# trivial methods to implement for the new type
Base.getproperty(crf::CalibratedRandomForestClassifier, s::Symbol) =
    if s in [:method, :rf, :cal]
        getfield(crf, s) # return a field of the struct type
    else
        getproperty(crf.rf, s) # return a property of the underlying PyObject
    end
ScikitLearn.is_classifier(::CalibratedRandomForestClassifier) = true
ScikitLearn.clone(crf::CalibratedRandomForestClassifier) =
    CalibratedRandomForestClassifier(crf.method, clone(crf.rf), clone.(crf.cal))
ScikitLearn.set_params!(crf::CalibratedRandomForestClassifier, args...; kwargs...) =
    set_params!(crf.rf, args...; kwargs...)

# implementation of fit! for instances of CalibratedRandomForestClassifier
function ScikitLearn.fit!(crf::CalibratedRandomForestClassifier,
                          X_train::Matrix,
                          y_train::Vector;
                          sample_weight::Vector=ones(length(y_train)))
    # fit the RandomForestClassifier
    ScikitLearn.fit!(crf.rf, X_train, y_train, sample_weight=sample_weight)
    
    # evaluate the OOB decision function
    oob_df = crf.rf.oob_decision_function_ # OOB function as returned by the classifier
    oob_i  = isfinite.(oob_df[:,1])        # examples for which oob_df is not NaN
    oob_df = oob_df[oob_i, :]              # select only those examples
    sample_weight = sample_weight[oob_i]
    Y = label_binarize(y_train[oob_i], 1:maximum(y_train[oob_i])) # one-hot encoding
    
    # fit one calibrator for each 1-vs-all problem
    crf.cal = map(k -> if crf.method == :isotonic
                           IsotonicRegression(out_of_bounds="clip")
                       else crf.method == :sigmoid
                           _SigmoidCalibration()
                       end, 1:size(Y, 2))
    for k in 1:length(crf.cal)
        ScikitLearn.fit!(crf.cal[k], oob_df[:,k], Y[:,k], sample_weight=sample_weight)
    end
    return crf # return fitted object
end

# implementation of predict_proba for instances of CalibratedRandomForestClassifier
function ScikitLearn.predict_proba(crf::CalibratedRandomForestClassifier, X_data::Matrix)
    df = ScikitLearn.predict_proba(crf.rf, X_data)
    p = reduce(hcat, map(k -> ScikitLearn.predict(crf.cal[k], df[:,k]), 1:length(crf.cal)))
    return p ./ sum(p, dims=2) # normalize (probabilities sum up to one)
end

# implementation of predict for instances of CalibratedRandomForestClassifier
function ScikitLearn.predict(crf::CalibratedRandomForestClassifier, X_data::Matrix)
    p = predict_proba(crf, X_data)
    return map(i -> findmax(p[i,:])[2], 1:size(p,1))
end

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

Configure a classifier to be used in `train_and_predict_proba`. The following values of
`classname` are available:

- GaussianNB
- RandomForestClassifier
- SVC (Support Vector Classifier)

`preprocessing` can be empty (default), or the name of a scikit-learn transformer like
`"StandardScaler"`.

The `calibration` can be `:none` (default), `:sigmoid`, or `:isotonic`.

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
    if classname == "CalibratedRandomForestClassifier" && calibration != :none
        classifier = Classifier(calibration; kwargs...)
    elseif calibration in [:sigmoid, :isotonic]
        classifier = CalibratedClassifierCV(
            classifier,
            method=string(calibration),
            cv=KFold(n_splits=3) # do not stratify CV
        )
    elseif calibration != :none
        throw(ArgumentError("calibration has to be :none, :sigmoid, or :isotonic"))
    end
    
    # add pre-processing
    if preprocessing != ""
        transformer = eval(Meta.parse(preprocessing))() # call the constructor method
        classifier  = Pipeline([ ("preprocessing", transformer), ("classifier", classifier) ])
    end
    
    return classifier
end


"""
    shuffle_split_subsample(X_full, y_full, X_auxtrain, y_auxtrain, f_train[; g=f,
                            seed, nobs = 50000, nobs_train = 100000)

Shuffle the `X_full` and `y_full`, split this data into training and observed data sets, and
subsample the training part with respect to the groups `g`. Three values for `f_train`
configure the kind of subsampling:

- `"appropriate"` means no subsampling and it thus maintains the inherent distribution of
  `X_full` and `y_full`.
- `"uniform"` undersamples a training set that is uniformly distributed in the target
  variable.
- `"auxiliary"` replaces the training part obtained by `X_full` and `y_full` with a shuffled
  subset of the auxiliary data provided with `X_auxtrain` and `y_auxtrain`.

**Returns**

- `X_data` the observed feature matrix
- `y_data` the target values of the observed data
- `X_train` the training feature matrix
- `y_train` the training labels

`nobs` and `nobs_train` optionally specify the maximum number of observations in these data
sets.
"""
function shuffle_split_subsample(
            X_full      :: AbstractMatrix,
            y_full      :: Vector{TM},
            X_auxtrain  :: Union{Nothing,Matrix{TN}},
            y_auxtrain  :: Union{Nothing,Vector{TM}},
            f_train     :: String;
            g           :: Vector{TO} = y_full,
            seed        :: Integer = convert(Int, rand(UInt32)),
            nobs        :: Int = 50000,
            nobs_train  :: Int = 100000,
            discretizer :: Union{Nothing,AbstractDiscretizer} = nothing ) where {TN<:Number, TM<:Number, TO<:Number}
    
    # shuffle and split
    urng   = MersenneTwister(seed)
    i_rand = randperm(urng, length(y_full))  # shuffled indices
    X_data = X_full[i_rand[1:nobs], :]        # first 50000 examples, by default
    y_data = y_full[i_rand[1:nobs]]
    X_train = X_full[i_rand[(nobs+1):end], :] # assume f_train=="appropriate" (original distribution)
    y_train = y_full[i_rand[(nobs+1):end]]
    g_train = g[i_rand[(nobs+1):end]]
    
    # subsample or replace by auxiliary training data
    if f_train == "uniform"
        if discretizer == nothing
            X_train, y_train = subsample_uniform(X_train, y_train; g=g_train) # discrete y_train
        else
            X_train, y_train = subsample_uniform(X_train, y_train, discretizer; g=g_train) # continuous y_train
        end
    elseif f_train == "auxiliary"
        i_aux   = randperm(urng, length(y_auxtrain)) # shuffle auxiliary data, too
        X_train = X_auxtrain[i_aux, :]
        y_train = y_auxtrain[i_aux]
    elseif f_train != "appropriate"
        throw(ArgumentException("'$(f_train)' is not a legal value for f_train"))
    end
    
    # by default, limit the training data to 100000 examples
    if nobs_train > 0
        nobs_train = min(length(y_train), nobs_train)
        X_train = X_train[1:nobs_train, :]
        y_train = y_train[1:nobs_train]
    end
    
    return X_data, y_data, X_train, y_train
end

"""
    subsample_uniform(X, y[, d;  g = y, shuffle = false])

Undersample a uniformly distributed data set consisting of the features matrix `X` and the
label array `y`, the last of which also defines the groups `g` by default.

If an AbstractDiscretizer `d` is specified, discretize the continuous values from `g` to
find its label while maintaining the continuous values of `g` and `y` in the result.
"""
function subsample_uniform( X :: AbstractMatrix{TN},
                            y :: Vector{TM};
                            g :: Vector{TI} = y,
                            shuffle::Bool=false ) where {TN<:Number, TM<:Number, TI<:Int}
    indices = subsample_uniform_indices(g, shuffle=shuffle)
    return convert(Matrix{TN}, X[indices, :]), convert(Vector{TM}, y[indices])
end

function subsample_uniform( X :: AbstractMatrix{TN},
                            y :: Vector{TM},
                            d :: AbstractDiscretizer;
                            g :: Vector{TO} = y,
                            shuffle::Bool=false ) where {TN<:Number, TM<:Number, TO<:Number}
    indices = subsample_uniform_indices(encode(d, g), shuffle=shuffle)
    return convert(Matrix{TN}, X[indices, :]), convert(Vector{TM}, y[indices])
end

"""
    subsample_uniform_indices(y[; shuffle = false])

Undersample a uniformly distributed data set with the label array `y`. Return the indices of
the sub-sample instead of the sub-sample itself.
"""
subsample_uniform_indices(y::Vector{TI}; shuffle::Bool=false) where TI<:Int =
    undersample(y, shuffle = shuffle).indices[1]

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

pairwise_mdpa(a::AbstractMatrix{T}) where T<:Number = pairwise_mdpa(a, a)
function pairwise_mdpa(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:Number
    mat = zeros(size(a, 2), size(b, 2))
    for i in 1:size(a, 2), j in i:size(b, 2)
        mat[i,j] = mdpa(a[:,i], b[:,j])
    end
    return full(Symmetric(mat))
end

"""
    chi2p(a, b)

Pearson's Chi Square distance between the pdfs `a` and `b`, where `b` is the truth.
"""
chi2p(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    Distances.wsqeuclidean(a, b, 1 ./ max.(b, DISTANCE_EPSILON))

"""
    kl(a, b)

Kullback-Leibler divergence between the pdfs `a` and `b`, where `b` is the truth.
"""
kl(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    Distances.kl_divergence(a, max.(b, DISTANCE_EPSILON)) # KL divergence only defined for b_i != 0

"""
    chi2s(a, b)

Symmetric Chi Square distance between the pdfs `a` and `b`.
"""
chi2s(a::AbstractVector{T}, b::AbstractVector{T}) where T<:Number =
    2 * Distances.chisq_dist(a, b)


"""
    bootstrap_sample_indices([rng, ] N)

Return the training set and test set indices for a data pool of size `N`.
"""
bootstrap_sample_indices(N::Int64) = bootstrap_sample_indices(Random.GLOBAL_RNG, N)
function bootstrap_sample_indices(rng::AbstractRNG, N::Int64)
    i_train = rand(rng, 1:N, N)
    return i_train, setdiff(1:N, i_train) # tuple of training and test set indices
end


"""
    latex_e(n[, digits = 2, dollars = true])

Format the number `n` in scientific LaTeX format.
"""
latex_e(n::Float64, digits::Int=2; dollars::Bool=true) =
    if n != 0
        m = match(r"(\d\.\d+)e(-?\d+)", @sprintf("%e", n))
        b = round(Base.parse(Float64, m[1]), digits=digits)
        s = (b == 1.0 ? "" : "$b \\cdot ") * "10^{$(m[2])}"
        return dollars ? "\$$s\$" : s
    else
        return "0"
    end


"""
    binarytransfer(df, y; normalize=true)

Empirically estimate the transfer matrix from the `y` column in the DataFrame `df` to the
item index. The resulting normalized matrix `R` can be used to deconvolve in a next-neighbor
fashion.

**Currently unused function for future work**
"""
function binarytransfer(df::DataFrame, y::Symbol;
                        ylevels::AbstractVector{T} = sort(unique(df[y])),
                        normalize::Bool = true)  where T<:Number # TODO change interface to Array types
    # construct binary matrix
    onehot = y_val::Float64 -> _diracimpulse(findfirst(ylevels, y_val), length(ylevels))
    R = reshape(vcat([ onehot(y_val) for y_val in df[y] ]...), (length(ylevels), size(df,1)))'
    
    if normalize
        return normalizetransfer!(R)
    else
        return R
    end
    
end

"""
    nextneighbor_pdf(data, train, features = propertynames(df); batchsize = 100)

A histogram of next neighbors to be used together with the matrix returned by
`Util.binarytransfer()`.

**Caution:** Throws an `OutOfMemoryError` when the batch size is too large.

**Currently unused function for future work**
"""
function nextneighbor_pdf(data::DataFrame, train::DataFrame,
                          features::Vector{Symbol} = propertynames(df);
                          batchsize::Int64 = 100) # TODO change interface to Array types
    data  = data[:,  features]
    train = train[:, features]
    g = zeros(Int64, size(train, 1))
    for batch in batchview(data, size = batchsize)
        for i in _findnextneighbors(batch, train)
            g[i] += 1
        end
    end
    return g
end

_findnextneighbors(data::AbstractDataFrame, train::AbstractDataFrame) =
    mapslices(col -> findmin(col)[2],
              pairwise(Euclidean(), convert(Array, data)', convert(Array, train)'), 2)

# Return a vector of length `l` that is `1.0` at position `p` and `0.0` in all other dimensions.
_diracimpulse(p::Int, l::Int) = (d = zeros(l);  d[p] = 1.0;  d)

end
