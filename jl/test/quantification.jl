# 
# Run these tests by calling include("test/quantification.jl")
# 
using Test
using DeconvExp
using DeconvExp.MoreMethods: OQT, ARC
using CherenkovDeconvolution: deconvolve
using Random, Distributions, Discretizers
using ScikitLearn: @sk_import
@sk_import datasets:make_blobs
@sk_import datasets:make_classification

include("../pyoqt.jl")
include("../pyARC.jl")

@testset "OQT PyOQT equal check with ordinal data" begin
    seed = 876
    @test begin
        d = Data.dataset("abalone")
        disc = Data.discretizer(d)
        X = Data.X_data(d)
        y = Data.y_data(d)
        y = encode(disc, y)
        clf = Util.DecisionTreeClassifier(random_state=seed)
        clf2 = clone(clf)
        f_pyoqt = deconvolve(PyOQT(clf; seed=seed), X, X, y)
        f_oqt = deconvolve(OQT(clf2; seed=seed), X, X, y)
        f_pyoqt ≈ f_oqt
    end
    @test begin
        d = Data.dataset("vote-for-clinton")
        disc = Data.discretizer(d)
        X = Data.X_data(d)
        y = Data.y_data(d)
        y = encode(disc, y)
        clf = Util.DecisionTreeClassifier(random_state=seed)
        clf2 = clone(clf)
        f_pyoqt = deconvolve(PyOQT(clf; seed=seed), X, X, y)
        f_oqt = deconvolve(OQT(clf2; seed=seed), X, X, y)
        f_pyoqt ≈ f_oqt
    end
end

@testset "OQT PyOQT equal check with artifical data" begin
    seed = 876
    @test begin
        @info "2 classes, 100 samples"
        X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2, n_classes = 2)
        y .+= 1
        clf = Util.DecisionTreeClassifier(random_state=seed)
        clf2 = clone(clf)
        @info "PyOQT"
        @time f_pyoqt = deconvolve(PyOQT(clf; seed = 876), X, X, y)
        @info "OQT"
        @time f_oqt = deconvolve(OQT(clf2; seed = 876), X, X, y)
        @info "- - - - - - - - - - - - -"
        f_pyoqt ≈ f_oqt
    end
    @test begin
        @info "2 classes, 10000 samples"
        X, y = make_classification(n_samples = 100000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 2)
        y .+= 1
        clf = Util.DecisionTreeClassifier(random_state=seed)
        clf2 = clone(clf)
        @info "PyOQT"
        @time f_pyoqt = deconvolve(PyOQT(clf; seed = 876), X, X, y)
        @info "OQT"
        @time f_oqt = deconvolve(OQT(clf2; seed = 876), X, X, y)
        @info "- - - - - - - - - - - - -"
        f_pyoqt ≈ f_oqt
    end
    @test begin
        @info "20 classes, 1000 samples"
        X, y = make_classification(n_samples = 1000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 20)
        y .+= 1
        clf = Util.DecisionTreeClassifier(random_state=seed)
        clf2 = clone(clf)
        @info "PyOQT"
        @time f_pyoqt = deconvolve(PyOQT(clf; seed = 876), X, X, y)
        @info "OQT"
        @time f_oqt = deconvolve(OQT(clf2; seed = 876), X, X, y)
        @info "- - - - - - - - - - - - -"
        f_pyoqt ≈ f_oqt
    end
end

@testset "ARC check" begin
    prepare_pseudo_topics(X,y) = begin
        n_samples = length(y)
        topics = vcat(fill("A", n_samples), fill("B", n_samples))
        (vcat(X,X), vcat(y,y), topics)
    end
    seed = 876
    @test begin 
        X_train, y_train = make_classification(n_samples = 1000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 5)
        y_train .+= 1
        X_train_sem, y_train_sem, topics_train = prepare_pseudo_topics(X_train, y_train)
        X_test, y_test = make_classification(n_samples = 1000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 5)
        y_test .+= 1
        base_clf = Util.DecisionTreeClassifier(random_state=seed)
        semevalQuantifier = py"RegressionQuantifier"(py"BinaryTreeRegressor"(clone(base_clf)))
        semevalQuantifier.fit(X_train_sem,y_train_sem,topics_train)
        f_semeval = semevalQuantifier.predict(X_test, fill("A", length(y_test)))["A"][2]
        f_arc = deconvolve(ARC(base_clf), X_test, X_train, y_train)
        # @info "f_semeval = $(f_semeval)"
        # @info "f_arc = $(f_arc)"
        isapprox(f_semeval, f_arc, atol=0.01)
    end
    @test begin 
        X_train, y_train = make_classification(n_samples = 10000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 10)
        y_train .+= 1
        X_train_sem, y_train_sem, topics_train = prepare_pseudo_topics(X_train, y_train)
        X_test, y_test = make_classification(n_samples = 10000, n_features = 100, n_informative = 40, n_redundant = 2, n_classes = 10)
        y_test .+= 1
        base_clf = Util.DecisionTreeClassifier(random_state=seed)
        semevalQuantifier = py"RegressionQuantifier"(py"BinaryTreeRegressor"(clone(base_clf)))
        semevalQuantifier.fit(X_train_sem,y_train_sem,topics_train)
        f_semeval = semevalQuantifier.predict(X_test, fill("A", length(y_test)))["A"][2]
        f_arc = deconvolve(ARC(base_clf), X_test, X_train, y_train)
        # @info "f_semeval = $(f_semeval)"
        # @info "f_arc = $(f_arc)"
        isapprox(f_semeval, f_arc, atol=0.01)
    end
end






