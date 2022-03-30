using CherenkovDeconvolution, OrdinalQuantification, PyCall, ScikitLearn

include("../pyoqt.jl")

struct OldPrefittedOQT <: DeconvolutionMethod
    root :: MoreMethods.TreeNode
    OldPrefittedOQT(x::MoreMethods.PrefittedOQT) = new(x.root)
end

# old implementation of predict_ordinal used in deconvolve
__numpy = pyimport_conda("numpy", "numpy")
old_predict_ordinal(n::MoreMethods.TreeNode, X::AbstractMatrix) = vcat(map(x -> old_predict_ordinal(n, x)', eachrow(X))...)
function old_predict_ordinal(n::MoreMethods.InnerNode, x::AbstractVector, p::Float64=1.0)
    p_r, p_l = ScikitLearn.predict_proba(n.h, reshape(x, (1, length(x))))
    return vcat(old_predict_ordinal(n.l, x, p_l), old_predict_ordinal(n.r, x, p_r)) .* p
end
old_predict_ordinal(n::MoreMethods.LeafNode, x::AbstractVector, p::Float64=1.0) = [p]
function old_predict_ordinal(n::MoreMethods.TreeNode, X::PyObject)
    shape = __numpy.shape(X)
    if length(shape) == 2
        return vcat(map(i -> __old_predict_ordinal_pyarray(n, X[i-1, :])', 1:shape[1])...)
    else
        throw(ArgumentError("length(__numpy.shape(X)) != 2"))
    end
end
function __old_predict_ordinal_pyarray(n::MoreMethods.InnerNode, x::PyObject, p::Float64=1.0)
    p_r, p_l = ScikitLearn.predict_proba(n.h, __numpy.reshape(x, (1, -1)))
    return vcat(
        __old_predict_ordinal_pyarray(n.l, x, p_l),
        __old_predict_ordinal_pyarray(n.r, x, p_r)
    ) .* p
end
__old_predict_ordinal_pyarray(n::MoreMethods.LeafNode, x::PyObject, p::Float64=1.0) = [p]

CherenkovDeconvolution.deconvolve(oqt::OldPrefittedOQT, X_obs::Any) =
    vec(mean(old_predict_ordinal(oqt.root, X_obs), dims=1))

# read and split the data
Random.seed!(123)
dataset = Data.Fact()
discr = Data.discretizer(dataset)
X_full = Data.X_data(dataset)
y_full = Data.y_data(dataset) # continuous values, not indices!
i_full = randperm(length(y_full))
i_trn = i_full[1:2000]
i_tst = i_full[2000:end]
X_trn = X_full[i_trn, :]
y_trn = encode(discr, y_full[i_trn])
X_tst = X_full[i_tst, :]
y_tst = encode(discr, y_full[i_tst])

clf = Util.DecisionTreeClassifier(random_state=1)

# test on dense data
for n_tst in [ 1000, 10000 ]
    s_py = @timed deconvolve(PyOQT(clf; seed=1), X_tst[1:n_tst,:], X_trn, y_trn)
    s_jl = @timed deconvolve(OQT(clf; seed=1), X_tst[1:n_tst,:], X_trn, y_trn)
    s_old = @timed deconvolve(OldPrefittedOQT(prefit(OQT(clf; seed=1), X_trn, y_trn)), X_tst[1:n_tst,:])
    f_py = s_py.value
    f_jl = s_jl.value
    @info "Result for n_tst=$(n_tst)" f_py ≈ f_jl t_py=s_py.time t_jl=s_jl.time t_old=s_old.time
end

# test on sparse data
scipy_sparse = pyimport("scipy.sparse")
X_trn_sparse = scipy_sparse.csr_matrix(X_trn)
X_tst_sparse = scipy_sparse.csr_matrix(X_tst)
for n_tst in [ 1000, 10000 ]
    s_py = @timed deconvolve(PyOQT(clf; seed=1), X_tst_sparse[1:n_tst,:], X_trn_sparse, y_trn)
    s_jl = @timed deconvolve(OQT(clf; seed=1), X_tst_sparse[1:n_tst,:], X_trn_sparse, y_trn)
    s_old = @timed deconvolve(OldPrefittedOQT(prefit(OQT(clf; seed=1), X_trn_sparse, y_trn)), X_tst_sparse[1:n_tst,:])
    f_py = s_py.value
    f_jl = s_jl.value
    @info "Result for n_tst=$(n_tst)" f_py ≈ f_jl t_py=s_py.time t_jl=s_jl.time t_old=s_old.time
end
