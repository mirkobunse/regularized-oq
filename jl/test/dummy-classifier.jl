#
# Run these tests by calling include("test/dummy-classifier.jl")
#
using Pkg
Pkg.activate(".")
using DeconvExp

Pkg.activate("test") # activate test/Project.toml
using Test

function TestCase()
    x = collect(1:8) ./ 9
    X = hcat(x, x .+ 1, x .+ 2)
    y = [ 0, 0, 0, 0, 1, 1, 2, 2 ]
    return Util.DummyClassifier(; X_trn=X, y_trn=y), collect(0:7), X, y
end

@testset "Do not change multi-class inputs" begin
    c, x_i, X, y = TestCase()
    c.fit(x_i, y)
    @test c.predict_proba(X) == X

    X_2 = rand(8, 3)
    @test c.predict_proba(X_2) == X_2

    X_3 = rand(100, 3)
    @test c.predict_proba(X_3) == X_3
end

@testset "Sum training inputs for binary decompositions" begin
    c, x_i, X, y = TestCase()
    y = Int.(y .== 0) # positive class
    c.fit(x_i, y)
    X_expected = hcat(vec(sum(X[:, 2:3], dims=2)), X[:, 1])
    @test c.predict_proba(X) == X_expected

    c, x_i, X, y = TestCase()
    y = Int.(y .== 1)
    c.fit(x_i, y)
    X_expected = hcat(vec(sum(X[:, [1, 3]], dims=2)), X[:, 2])
    i = randperm(8)
    X = X[i, :]
    X_expected = X_expected[i, :]
    @test c.predict_proba(X) == X_expected

    c, x_i, X, y = TestCase()
    i = randperm(8)
    X = X[i, :]
    y = y[i]
    c = Util.DummyClassifier(; X_trn=X, y_trn=y) # reorder requires re-initialization
    y = Int.(y .== 2)
    c.fit(x_i, y)
    X_expected = hcat(vec(sum(X[:, 1:2], dims=2)), X[:, 3])
    @test c.predict_proba(X) == X_expected
end

@testset "Sum arbitrary inputs for binary decompositions" begin
    c, x_i, X, y = TestCase()
    y = Int.(y .== 0) # positive class
    c.fit(x_i, y)
    X = rand(8, 3)
    X_expected = hcat(vec(sum(X[:, 2:3], dims=2)), X[:, 1])
    @test c.predict_proba(X) == X_expected

    c, x_i, X, y = TestCase()
    y = Int.(y .== 1)
    c.fit(x_i, y)
    X = rand(100, 3)
    X_expected = hcat(vec(sum(X[:, [1, 3]], dims=2)), X[:, 2])
    i = randperm(8)
    X = X[i, :]
    X_expected = X_expected[i, :]
    @test c.predict_proba(X) == X_expected

    c, x_i, X, y = TestCase()
    i = randperm(8)
    X = X[i, :]
    y = y[i]
    c = Util.DummyClassifier(; X_trn=X, y_trn=y) # reorder requires re-initialization
    y = Int.(y .== 2)
    c.fit(x_i, y)
    X = rand(100, 3)
    X_expected = hcat(vec(sum(X[:, 1:2], dims=2)), X[:, 3])
    @test c.predict_proba(X) == X_expected
end

@testset "Subsample in binary decompositions" begin
    c, x_i, X, y = TestCase()
    x_i = [ 1, 5, 7 ] # one instance of each class
    X = X[x_i, :]
    y = y[x_i]
    y = Int.(y .== 0) # positive class
    c.fit(x_i, y)
    X_expected = hcat(vec(sum(X[:, 2:3], dims=2)), X[:, 1])
    @test c.predict_proba(X) == X_expected
end
