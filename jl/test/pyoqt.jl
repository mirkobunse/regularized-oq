
using PyCall, Random, StatsBase, CherenkovDeconvolution, OrdinalQuantification.Util, ScikitLearn
using OrdinalQuantification.MoreMethods: fit_ordinal

function _init_pyOQT()

    py"""
    import numpy as np
    import sys
    import math

    def pcc(fname, indmin, indmax):
        with open(fname, "r") as f:
            lines = f.readlines()
            lines = lines[indmin:indmax]
            col=2
            if lines[0].split(" ")[1]=="-1":
                col=1
            return sum([ float(l.split(" ")[col]) for l in lines[1:] ])/float(len(lines)-1)

    def classify(probs, ar):
        res = []
        for i in range(len(ar)):
            s=1.0
            for j,k in zip(probs,ar[i]):
                if k>0:
                    s*=j*k
            res.append(s)
        return res

    def ordinalClassification(classifications, oqtree):
        fres=[ [] for x in range(len(oqtree)) ]
        for i in range(len(classifications[0])):
            r=classify([ classifications[j][i] for j in range(len(classifications)) ], oqtree)
            for i,x in enumerate(r):
                fres[i].append(x)
        return [ float(sum(x))/len(x) for x in fres ]

    def initMatrix(x,y):
        return [[0 for i in range(y)] for i in range(x)]

    def elementByElementMatrixProduct(m1,m2):
        m=[ [0]*(len(m1[0])) ]*(len(m1))
        for x in range(len(m1)):
            for y in range(len(m1[x])):
                m[x][y]=m1[x][y]*m2[x][y]
        return m

    def elementByElementMatrixXOR(m1,m2):
        m=initMatrix(len(m1), len(m1[0]))
        for x in range(len(m1)):
            for y in range(len(m1[x])):
                if m1[x][y]==1 or m2[x][y]==1:
                    m[x][y]=1
        return m

    def argmin(l):
        if len(l)==0:
            return None
        return sorted([ (y,x) for x,y in enumerate(l) ])[0][1]

    def buildOQTree(m, kldvec, classes, parent=None): 
        if parent!=None:
            for x in classes:
                m[x][parent]=1
        x=argmin(kldvec)
        if kldvec[x]==sys.float_info.max:
            return m
        kldvec[x]=sys.float_info.max
        return elementByElementMatrixXOR(buildOQTree(m,[ y if i<x else sys.float_info.max for i,y in enumerate(kldvec) ],[ y for y in classes if y<=x ],x*2),
                                         buildOQTree(m,[ y if i>x else sys.float_info.max for i,y in enumerate(kldvec) ],[ y for y in classes if y>x ],x*2+1))
        return m

    def computeKLDbinary(d, trued):
        KLD = []
        for x,y in zip(d, trued):
            k = 0.0
            if y > 0.0:
                k += y*math.log(y/x,2)
            if y < 1.0:
                k += (1-y)*math.log((1-y)/(1-x),2)
            KLD.append(k)
        return KLD
    """
end


"""
    PyOQT(classifier; kwargs...)

    Wrapper for ordinal quantification tree by Da San Martino et al. 
    
    **Keyword arguments**

    - `val_split = 0.4`
    is the fraction of the validation set for tree induction.
    - `seed = nothing`
    is an optional seed state to reproduce the validation splits
"""
mutable struct PyOQT <: DeconvolutionMethod
    classifier :: Any
    val_split :: Float64
    seed :: Union{Int64,Nothing}
    oqtree :: Matrix
    PyOQT(classifier; val_split::Float64=0.4, seed::Union{Int64,Nothing}=nothing) = begin
        _init_pyOQT()
        new(classifier, val_split, seed)
    end
end

function _build_oqtree_matrix(H::Any, X_val::Any, y_val::Vector)
    n_classifiers = length(H)
    n_classes = n_classifiers + 1
    goldLabelsCDF = cumsum(counts(y_val) ./ length(y_val))
    goldLabelsCDF = goldLabelsCDF[1:n_classifiers]
    pcc(j) = mean(H[j].predict_proba(X_val)[:, 2])
    pccBinaryClassifiers = map(j -> pcc(j), 1:n_classifiers)
    KLD = py"computeKLDbinary"(pccBinaryClassifiers, goldLabelsCDF)

    zeroMatrix = py"initMatrix"(n_classes, n_classifiers * 2)
    oqtree = py"buildOQTree"(zeroMatrix, KLD, 0:n_classes-1)
    oqtree
end

function _predict_pyoqt(H::Any, oqt::Matrix, X_obs::Any)
    n_classifiers = length(H)
    swap = f -> hcat(f[:, 2], f[:, 1])
    classifications = transpose(hcat([swap(H[j].predict_proba(X_obs)) for j = 1:n_classifiers]...))
    f_est = py"ordinalClassification"(classifications, oqt)
    DeconvUtil.normalizepdf!(f_est)
end

CherenkovDeconvolution.deconvolve(
    pyOqt::PyOQT,
    X_obs::Any,
    X_trn::Any,
    y_trn::AbstractVector{I}
) where {T,N,I<:Integer} =
    deconvolve(pyOqt, X_obs, X_trn, convert(Vector, y_trn))

function CherenkovDeconvolution.deconvolve(
    pyOqt::PyOQT,
    X_obs::Any,
    X_trn::Any,
    y_trn::Vector{I}
) where {T,N,I<:Integer}
    H, X_val, y_val,_ = fit_ordinal(pyOqt.classifier, X_trn, y_trn, pyOqt.val_split; seed=pyOqt.seed)
    pyOqt.oqtree = _build_oqtree_matrix(H, X_val, y_val)
    _predict_pyoqt(H, pyOqt.oqtree, X_obs)
end










