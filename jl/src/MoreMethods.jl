module MoreMethods

using
    CherenkovDeconvolution,
    Discretizers,
    Distances,
    Downloads,
    LinearAlgebra,
    Optim,
    PyCall,
    Random,
    ScikitLearn,
    StatsBase,
    QuaPy,
    QUnfold,
    OrderedCollections
import Conda

# download and import the original implementation of the BinaryTreeRegressor
const __numpy = PyNULL()
const __castano_factory = PyNULL()
const __castano_emd_distances = PyNULL()
const __sklearn_euclidean_distances = PyNULL()
const __BinaryTreeRegressor = PyNULL()
function __init__()
    copy!(__numpy, pyimport_conda("numpy", "numpy"))

    castano_factory = pyimport_e("ordinal_quantification.factory")
    if ispynull(castano_factory) # need to install ordinal_quantification?
        Conda.pip_interop(true)
        Conda.pip("install", "git+https://github.com/mirkobunse/ordinal_quantification.git")
        castano_factory = pyimport("ordinal_quantification.factory")
    end
    copy!(__castano_factory, castano_factory)
    copy!( # we can now safely assume that ordinal_quantification is installed
        __castano_emd_distances,
        pyimport("ordinal_quantification.metrics.ordinal").emd_distances
    )
    copy!( # we can now safely assume that ordinal_quantification is installed
        __sklearn_euclidean_distances,
        pyimport("sklearn.metrics.pairwise").euclidean_distances
    )

    python_file = "deps/binary_tree_regressor.py"
    if !isfile(python_file)
        # download and repair the original implementation
        url = "https://raw.githubusercontent.com/aesuli/semeval2016-task4/8d0c6f2365e29efa7bb08b1521299c6e3a875ed0/binary_tree_regressor.py"
        mkpath(dirname(python_file))
        Downloads.download(url, python_file)
        open(python_file, "w") do io
            for line in readlines(download(url))
                line = replace(line,
                    "self._predict(x, self._fitted_estimator)" =>
                    "self._predict(x if len(np.shape(x)) == 2 else [x], self._fitted_estimator)"
                )
                line = replace(line, "return y_pred" => "return np.array(y_pred)")
                println(io, line)
                if line == "        self._fitted_estimator = self._fit(X, y)"
                    println(io, "        self.classes_ = np.unique(y)")
                    println(io, "        self.n_classes_ = len(self.classes_)")
                end
            end
        end
    end
    python_path = pyimport("sys")."path"
    if "deps" ∉ python_path
        pushfirst!(python_path, "deps")
    end
    copy!(__BinaryTreeRegressor, pyimport("binary_tree_regressor").BinaryTreeRegressor)
end
BinaryTreeRegressor(clf::Any; verbose::Bool=false) = __BinaryTreeRegressor(clf, verbose)

# ==============
# QuaPy wrappers
# ==============

struct QuaPyMethod <: DeconvolutionMethod
    quantifier::QuaPy.Methods.BaseQuantifier
    fit_learner::Bool
end

# meta-programming: generate constructors for all QuaPy methods, e.g. ClassifyAndCount
for m in vcat(values(QuaPy.Methods.__METHODS)...)
    @eval $(m)(args...; fit_learner::Bool=true, kwargs...) =
        QuaPyMethod(QuaPy.Methods.$(m)(args...; kwargs...), fit_learner)
end

reconfigure(m::QuaPyMethod; fit_learner::Bool=true) =
    QuaPyMethod(m.quantifier, fit_learner)

CherenkovDeconvolution.deconvolve(
        m::QuaPyMethod,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer} =
    deconvolve(prefit(m, X_trn, y_trn), X_obs)

CherenkovDeconvolution.deconvolve(m::QuaPyMethod, X_obs::Any) =
    quantify(m.quantifier, X_obs)

function CherenkovDeconvolution.prefit(m::QuaPyMethod, X::Any, y::AbstractVector{I}) where {I<:Integer}
    if hasproperty(m.quantifier.__object, :val_split) # reproducible validation splits
        X_trn, y_trn, X_val, y_val = split_validation(m.quantifier.__object.val_split, X, y)
        val_lc = QuaPy.Datasets.LabelledCollection(X_val, y_val)
        QuaPy.fit!(m.quantifier, X_trn, y_trn; val_split=val_lc.__object, fit_learner=m.fit_learner)
    else
        QuaPy.fit!(m.quantifier, X, y; fit_learner=m.fit_learner)
    end
    return m
end

function split_validation(v::Real, X::Any, y::AbstractVector{I}) where {I<:Integer}
    if v == 1.0 # use all data for training AND validation
        i_trn = collect(1:length(y))
        i_val = i_trn
    else
        p = DeconvUtil.fit_pdf(y)
        i_trn, i_val = pyimport("sklearn.model_selection").train_test_split(
            collect(1:length(y)); # split indices
            test_size = v,
            stratify = y,
            random_state = rand(UInt32)
        )
    end
    X_trn = X[isa(X, PyObject) ? i_trn .- 1 : i_trn, :] # Python indices start at 0
    y_trn = y[i_trn]
    X_val = X[isa(X, PyObject) ? i_val .- 1 : i_val, :]
    y_val = y[i_val]
    return X_trn, y_trn, X_val, y_val
end

# ============================================================
# QuaPy extensions (might require the master branch of QuaPy!)
# ============================================================
OrdinalEMQ(classifier::Any; fit_learner::Bool=true, smoothing::Smoothing=NoSmoothing()) =
    EMQ(classifier; fit_learner=fit_learner, transform_prior=f_est->apply(smoothing, f_est))

# ===================================================================================
# QUnfold wrappers
# ===================================================================================

CherenkovDeconvolution.deconvolve(
        m::T,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {T<:QUnfold.AbstractMethod, I<:Integer} =
    QUnfold.predict(QUnfold.fit(m, X_trn, y_trn), X_obs)

CherenkovDeconvolution.prefit(
        m::T,
        X_trn::Any,
        y_trn::Vector{I}
        ) where {T<:QUnfold.AbstractMethod, I<:Integer} =
    QUnfold.fit(m, X_trn, y_trn)

CherenkovDeconvolution.deconvolve(m::T, X_obs::Any) where {T<:Union{QUnfold.FittedMethod,QUnfold._FittedEDX}}=
    QUnfold.predict(m, X_obs)

# enable EarthMoversSurrogate as a EDy distance
function (d::EarthMoversSurrogate{Cityblock})(
        a::AbstractVector{T},
        b::AbstractVector{T}
        ) where {T<:Number}
    if length(a) != length(b)
        error("length(a) = $(length(a)) != length(b) = $(length(b))")
    elseif !isapprox(sum(a), sum(b))
        error("sum(a) = $(sum(a)) != sum(b) = $(sum(b))")
    end
    prefixsum = 0.0 # algorithm 1 in [cha2002measuring]
    distance  = 0.0
    for i in 1:length(a)
        prefixsum += a[i] - b[i]
        distance  += prefixsum^2
    end
    return distance / sum(a) # this normalization renders MDPA equivalent to EMD
end
(d::EarthMoversSurrogate{Cityblock})(a::T, b::T) where {T<:Number} = (a - b)^2

# ===================================================================================
# https://github.com/mirkobunse/ordinal_quantification wrappers
# ===================================================================================

struct _CastanoMethod <: DeconvolutionMethod
    quantifier::PyObject
end

CherenkovDeconvolution.deconvolve(
        m::_CastanoMethod,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer} =
    deconvolve(prefit(m, X_trn, y_trn), X_obs)

CherenkovDeconvolution.deconvolve(m::_CastanoMethod, X_obs::Any) =
    try # do not warn about zero entries
        return DeconvUtil.normalizepdf(m.quantifier.predict(X_obs); warn=false)
    catch any_exception # print backtrace and return a uniform estimate
        Base.printstyled("\nWARNING (_CastanoMethod): "; color=:yellow, bold=true)
        Base.showerror(stdout, any_exception)
        Base.show_backtrace(stdout, Base.catch_backtrace())
        C = length(m.quantifier.estimator_test.classes_)
        return ones(C) ./ C
    end

function CherenkovDeconvolution.prefit(m::_CastanoMethod, X::Any, y::AbstractVector{I}) where {I<:Integer}
    m.quantifier.fit(X, y)
    return m
end

CastanoCC(classifier::Any; kwargs...) =
    _CastanoMethod(__castano_factory.CC(classifier; kwargs...))
CastanoAC(classifier::Any; kwargs...) =
    _CastanoMethod(__castano_factory.AC(classifier; kwargs...))
CastanoPCC(classifier::Any; kwargs...) =
    _CastanoMethod(__castano_factory.PCC(classifier; kwargs...))
CastanoPAC(classifier::Any; kwargs...) =
    _CastanoMethod(__castano_factory.PAC(classifier; kwargs...))
CastanoEDy(classifier::Any; kwargs...) =
    _CastanoMethod(__castano_factory.EDy(classifier; kwargs...))
CastanoPDF(classifier::Any, n_bins; kwargs...) =
    _CastanoMethod(__castano_factory.PDF(classifier, n_bins; kwargs...))

# ===================================================================================
# Numerically Adjusted Classify & Count (NACC) with proper losses and regularizations
#
# The numerical adjustment is used to implement o-ACC and o-PACC.
# ===================================================================================
struct ClassificationBinning <: Binning
    classifier::Any
end
struct ClassificationDiscretizer <: BinningDiscretizer
    classifier::Any
    J::Int
end
function CherenkovDeconvolution.BinningDiscretizer(
        b::ClassificationBinning,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer}
    y_trn = y_trn .+ (1 - minimum(y_trn)) # set  minimum of y_trn to 1
    ScikitLearn.fit!(b.classifier, X_trn, y_trn)
    return ClassificationDiscretizer(b.classifier, length(unique(y_trn)))
end
CherenkovDeconvolution.Binnings.bins(b::ClassificationDiscretizer) = collect(1:b.J)
Discretizers.encode(d::ClassificationDiscretizer, X_obs::Any) =
    ScikitLearn.predict(d.classifier, X_obs)

struct _NACC <: CherenkovDeconvolution.DiscreteMethod
    classifier::Any
    criterion::Symbol
    regularization::Symbol
    tau::Float64
    epsilon::Float64
    K::Int
    val_split::Float64
    n_bins_y::Int
    is_probabilistic::Bool
end
OrdinalACC(classifier::Any;
        criterion::Union{Symbol,AbstractString}=:mse,
        regularization::Union{Symbol,AbstractString}=:curvature,
        tau::Float64=0.0,
        epsilon::Float64=1e-12,
        K::Int=10_000,
        val_split::Float64=0.334,
        n_bins_y::Int=-1) =
    _NACC(classifier, Symbol(criterion), Symbol(regularization), tau, epsilon, K, val_split, n_bins_y, false)
OrdinalPACC(classifier::Any;
        criterion::Union{Symbol,AbstractString}=:mse,
        regularization::Union{Symbol,AbstractString}=:curvature,
        tau::Float64=0.0,
        epsilon::Float64=1e-12,
        K::Int=10_000,
        val_split::Float64=0.334,
        n_bins_y::Int=-1) =
    _NACC(classifier, Symbol(criterion), Symbol(regularization), tau, epsilon, K, val_split, n_bins_y, true)
CherenkovDeconvolution.Methods.expects_normalized_R(m::_NACC) = true
CherenkovDeconvolution.Methods.expected_n_bins_y(m::_NACC) = m.n_bins_y

function CherenkovDeconvolution.prefit(m::_NACC, X::Any, y::AbstractVector{I}) where {I<:Integer}
    n_bins_y = max(
        CherenkovDeconvolution.Methods.expected_n_bins_y(m),
        CherenkovDeconvolution.Methods.expected_n_bins_y(y)
    ) # number of classes/bins
    try # see CherenkovDeconvolution.prefit(m::DiscreteMethod, ...)
        check_arguments(X, y)
    catch exception
        if isa(exception, LoneClassException)
            f_est = recover_estimate(exception, n_bins_y)
            @warn "Only one label in the training set, returning a trivial estimate" f_est
            return f_est
        else
            rethrow()
        end
    end
    label_sanitizer = LabelSanitizer(y, n_bins_y)
    y = encode_labels(label_sanitizer, y) # encode labels for safety

    # split the data into an actual training set and a validation set
    X_trn, y_trn, X_val, y_val = split_validation(m.val_split, X, y)
    d = BinningDiscretizer(ClassificationBinning(m.classifier), X_trn, y_trn)

    # estimate the transfer matrix R
    if m.is_probabilistic
        y_pred = ScikitLearn.predict_proba(d.classifier,X_val)
        R = LinearAlgebra.diagm(ones(n_bins_y))
        for i in 1:n_bins_y
            R[:, i] = mean(y_pred[y_val.==i, :]; dims=1)
        end
    else # count co-occurences on a val_split fraction of the data
        R = DeconvUtil.fit_R(y_val, encode(d, X_val); bins_x=CherenkovDeconvolution.Binnings.bins(d))
    end
    f_trn = DeconvUtil.fit_pdf(y)
    return PrefittedMethod(m, d, R, label_sanitizer, f_trn, f_trn)
end

function CherenkovDeconvolution.deconvolve(prefitted_m::PrefittedMethod{_NACC}, X_obs::Any)
    m = prefitted_m.method
    R = prefitted_m.R
    label_sanitizer = prefitted_m.label_sanitizer
    f_trn = prefitted_m.f_trn
    f_0 = prefitted_m.f_0

    # estimate the observed density function g
    if m.is_probabilistic
        g = vec(mean(ScikitLearn.predict_proba(
            prefitted_m.discretizer.classifier,
            X_obs
        ); dims=1))
    else
        g = DeconvUtil.fit_pdf(
            encode(prefitted_m.discretizer, X_obs),
            bins(prefitted_m.discretizer)
        )
    end

    CherenkovDeconvolution.Methods.check_discrete_arguments(R, g)
    d = length(f_trn)

    # set up the regularizer and the loss function
    C = if m.regularization == :curvature
        LinearAlgebra.diagm(
            -1 => fill(-1, d-1),
            0 => fill(2, d),
            1 => fill(-1, d-1)
        )[2:(d-1), :]
    elseif m.regularization == :difference
        LinearAlgebra.diagm(
            0 => fill(1, d),
            1 => fill(-1, d-1)
        )[1:(d-1), :]
    elseif m.regularization == :norm
        LinearAlgebra.diagm(ones(d))
    else
        throw(ArgumentError("Unknown regularization $(m.regularization)"))
    end
    objective = if m.criterion == :mse
        f -> sum((g - R*f).^2) + m.tau * sum((C*f).^2)
    elseif m.criterion == :mae
        f -> sum(abs.(g - R*f)) + m.tau * sum((C*f).^2)
    else
        throw(ArgumentError("Unknown criterion $(m.criterion)"))
    end

    # optimize the regularized loss
    conf = Optim.Options(
        g_tol = m.epsilon,
        iterations = m.K # maximum number of iterations
    )
    res = optimize(objective, f_0, BFGS(), conf; autodiff=:forward)
    f_est = Optim.minimizer(res)
    if !Optim.converged(res)
        @warn "NACC did not converge; results may be unsatisfactory"
    end
    @debug "NACC" Optim.minimum(res) Optim.iterations(res)

    return DeconvUtil.normalizepdf(decode_estimate(label_sanitizer, f_est); warn=false)
end

# ===============================
# OrdinalQuantificationTree (OQT)
# ===============================

"""
    TreeNode(H, ℓ)

Generate an ordinal tree, consisting of `InnerNode` and `LeafNode` instances,
from a set of binary classifiers `H` and classifier-wise divergences or losses `ℓ`.
"""
abstract type TreeNode end
struct InnerNode <: TreeNode
    h :: Any # fitted classifier
    l :: TreeNode
    r :: TreeNode
end
struct LeafNode <: TreeNode end

function TreeNode(H::Vector{T}, ℓ::Vector{R}) where {T,R<:Real}
    if all(ℓ .== Inf) # no more splitting possible?
        return LeafNode()
    end
    t = findmin(ℓ)[2] # select the best threshold
    @debug "Splitting at threshold $t" ℓ

    # prevent future selections of t (left and right sides)
    ℓ_l = copy(ℓ)
    ℓ_r = copy(ℓ)
    ℓ_l[t:end] .= Inf # left side ignores the right side
    ℓ_r[1:t] .= Inf # right side ignores the left side

    # recursion
    return InnerNode(H[t], TreeNode(H, ℓ_l), TreeNode(H, ℓ_r))
end

predict_ordinal(n::TreeNode, X::AbstractMatrix) = predict_ordinal(n, X, ones(size(X, 1)))
predict_ordinal(n::TreeNode, X::PyObject) = predict_ordinal(n, X, ones(__numpy.shape(X)[1]))
function predict_ordinal(n::InnerNode, X::Any, p::Vector{Float64})
    proba = ScikitLearn.predict_proba(n.h, X)
    return hcat(
        predict_ordinal(n.l, X, proba[:,2]),
        predict_ordinal(n.r, X, proba[:,1])
    ) .* p
end
predict_ordinal(n::LeafNode, X::Any, p::Vector{Float64}) = p

"""
    OrdinalClassifier(classifier; val_split=0.4)

Meta-model of a base `classifier` which decomposes an ordinal classification task into
a sequence of thresholded binary tasks.
"""
mutable struct OrdinalClassifier
    classifier :: Any # hopefully implements the sklearn API
    val_split :: Float64
    root :: TreeNode
    OrdinalClassifier(c::Any; val_split::Float64=0.4) = new(c, val_split, LeafNode())
end

# fit a sequence of thresholded binary classifiers
function fit_ordinal(classifier::Any, X::Any, y::Vector, val_split::Float64)
    y = y .+ (1 - minimum(y)) # set  minimum of y_trn to 1
    n_classes = maximum(y) # TODO sanitize and check the arguments
    X_trn, y_trn, X_val, y_val = split_validation(val_split, X, y)

    # build classifiers
    H = map(1:(n_classes-1)) do j
        h_j = clone(classifier)
        ScikitLearn.fit!(h_j, X_trn, y_trn .<= j)
        h_j
    end
    return H, X_val, y_val, n_classes # return the classifiers and the validation split
end

# implementation of the sklearn API for OrdinalClassifiers
function ScikitLearn.fit!(clf::OrdinalClassifier, X::Any, y::Vector)
    H, X_val, y_val, n_classes = fit_ordinal(clf.classifier, X, y, clf.val_split)

    # compute the accuracies in each binary subtask; TODO allow other metrics
    ℓ = map(1:(n_classes-1)) do j
        y_true = y_val .<= j
        y_pred = ScikitLearn.predict(H[j], X_val)
        mean(y_true .== y_pred)
    end

    # build an ordinal classification tree
    clf.root = TreeNode(H, ℓ)
    return clf # return the fitted object
end
ScikitLearn.predict_proba(clf::OrdinalClassifier, X::Any) = predict_ordinal(clf.root, X)
function ScikitLearn.predict(clf::OrdinalClassifier, X::Any)
    p = ScikitLearn.predict_proba(clf, X)
    return map(i -> findmax(p[i,:])[2], 1:size(p,1))
end

"""
    OQT(classifier; kwargs...)

Ordinal quantification tree by Da San Martino et al.

**Keyword arguments**

- `epsilon = 0.0`
  is the smoothing factor for the KL divergence.
- `val_split = 0.4`
  is the fraction of the validation set for tree induction.
- `seed = nothing`
  is an optional seed state to reproduce the validation splits
"""
struct OQT <: DeconvolutionMethod
    classifier :: Any # hopefully implements the sklearn API
    epsilon :: Float64
    val_split :: Float64
    OQT(c;
        epsilon :: Float64 = 1e-6,
        val_split :: Float64 = 0.4,
        ) = new(c, epsilon, val_split)
end

struct PrefittedOQT <: DeconvolutionMethod
    root :: TreeNode
end

# ScikitLearn.jl goes mad when another sub-type of AbstractArray is used.
CherenkovDeconvolution.deconvolve(
        oqt::OQT,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer} =
    deconvolve(prefit(oqt, X_trn, convert(Vector, y_trn)), X_obs)

function CherenkovDeconvolution.prefit(
        oqt::OQT,
        X_trn::Any,
        y_trn::Vector{I}
        ) where {I<:Integer}
    # TODO sanitize and check the arguments

    # kullback–leibler divergence
    KLDbinary(p_pred, p_true) = begin
        k = 0.0
        if p_true > 0.0
            k += p_true * log(2, p_true / p_pred)
        end
        if p_true < 1.0
            k += (1 - p_true) * log(2, (1 - p_true) / (1 - p_pred))
        end
        k
    end

    # fit classifiers and evaluate the KL divergences in each binary sub-task
    H, X_val, y_val, n_classes = fit_ordinal(oqt.classifier, X_trn, y_trn, oqt.val_split)
    ℓ = map(1:(n_classes-1)) do j
        p_true = mean(y_val .<= j) # binary true distribution = [p_true, 1-p_true]
        p_pred = mean(ScikitLearn.predict_proba(H[j], X_val)[:,2]) # binary PCC
        p_true = (p_true+oqt.epsilon) / (1+2*oqt.epsilon) # smoothing
        p_pred = (p_pred+oqt.epsilon) / (1+2*oqt.epsilon)
        KLDbinary(p_pred, p_true)
    end

    # build an OQ-tree
    return PrefittedOQT(TreeNode(H, ℓ))
end

CherenkovDeconvolution.deconvolve(oqt::PrefittedOQT, X_obs::Any) =
    vec(mean(predict_ordinal(oqt.root, X_obs), dims=1))

# =======================
# AdjustedRegressAndCount
# =======================

"""
    AdjustedRegressAndCount(classifier; kwargs...)
    ARC(classifier; kwargs...)

The adjusted regress and count (ARC) method by Esuli.

**Keyword arguments**

- `with_binary_tree_regressor = true`
  decides whether the `classifier` is used as the base classifier of a
  `BinaryTreeRegressor` (default), or whether the predictions of the
  `classifier` are used directly.
  `val_split = 0.4`
  specifies the validation data to be used. 
"""
struct AdjustedRegressAndCount <: DeconvolutionMethod
    classifier :: Any
    with_binary_tree_regressor :: Bool
    val_split :: Float64
    AdjustedRegressAndCount(
        classifier::Any;
        with_binary_tree_regressor::Bool=true,
        val_split::Float64=0.4
    ) = new(classifier, with_binary_tree_regressor, val_split)
end
const ARC = AdjustedRegressAndCount # type alias

struct PrefittedARC{I<:Integer} <: DeconvolutionMethod
    classifier :: Any
    w :: Vector{Float64}
    labels :: Vector{I}
end

CherenkovDeconvolution.deconvolve(
        m::ARC,
        X_obs::Any,
        X_trn::Any,
        y_trn::AbstractVector{I}
        ) where {I<:Integer} =
    deconvolve(prefit(m, X_trn, convert(Vector, y_trn)), X_obs)

function CherenkovDeconvolution.prefit(m::ARC, X::Any, y::Vector{I}) where {I<:Integer}
    clf = if m.with_binary_tree_regressor
        BinaryTreeRegressor(m.classifier)
    else
        m.classifier
    end
    y = y .+ (1 - minimum(y)) # set  minimum of y_trn to 1
    X_trn, y_trn, X_val, y_val = split_validation(m.val_split, X, y)
    ScikitLearn.fit!(clf, X_trn, y_trn)
    labels = sort(unique(y))
    n_classes = length(labels)
    f_val = sort(proportionmap(y_val))
    f̂_val = sort(proportionmap(ScikitLearn.predict(clf, X_val)))

    # calculate correction vector w
    w = zeros(n_classes)
    for (i, class) in enumerate(labels)
        if class in keys(f_val) && class in keys(f̂_val)
            w[i] = f_val[class] / f̂_val[class]
        else
            w[i] = 0.0
        end
    end
    return PrefittedARC(clf, w, labels)
end

function CherenkovDeconvolution.deconvolve(m::PrefittedARC, X_obs::Any)
    f̂_obs = sort(proportionmap(ScikitLearn.predict(m.classifier, X_obs)))
    f̂_obs_corrected = zeros(length(m.labels))
    for (i, class) in enumerate(m.labels)
        if class in keys(f̂_obs)
            f̂_obs_corrected[i] = f̂_obs[class]
        end
        if m.w[i] > 0.0
            f̂_obs_corrected[i] *= m.w[i]
        else
            @warn "For class $(class) fallback to uncorrected solution."
        end
    end
    return DeconvUtil.normalizepdf!(f̂_obs_corrected; warn=false)
end

end # module
