using Distributions, Random, Test

@testset "Data.subsample_indices(...; min_samples_per_class=0, allow_duplicates=false)" begin
    for seed in 1:10
        # generate a random task
        rng = MersenneTwister(seed)
        n_classes = rand(rng, 3:10)
        n_samples = n_classes * rand(rng, 1000:10000)
        N = rand(rng, 500:2000)
        y = rand(rng, 1:n_classes, n_samples)
        @test length(unique(y)) == n_classes # check correctness

        # generate random prevalences where no probability is zero
        p = rand(rng, Dirichlet(ones(n_classes)))
        p = (p .+ 1/N) ./ ((N+n_classes)/N)
        p_min = (round.(Int, N * p) .- 1) / N # boundaries of valid solutions
        p_max = (round.(Int, N * p) .+ 1) / N

        # basic testing
        indices = Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=0, allow_duplicates=false)
        @test length(indices) == N
        @test length(unique(indices)) == N
        @test length(unique(y[indices])) == n_classes
        @test indices == Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=0, allow_duplicates=false)
        @test all(DeconvUtil.fit_pdf(y[indices], 1:n_classes) .>= p_min)
        @test all(DeconvUtil.fit_pdf(y[indices], 1:n_classes) .<= p_max)

        # "corrected" prevalences for which only one valid solution exists 
        p_corr = round.(Int, N * p) / N
        while N - sum(round.(Int, N * p_corr)) >= 1/N
            p_corr[rand(1:n_classes)] += 1/N
        end
        while sum(round.(Int, N * p_corr)) - N >= 1/N
            p_corr[rand(1:n_classes)] -= 1/N
        end
        @test sum(round.(Int, N * p_corr)) == N
        indices_corr = Data.subsample_indices(rng, y, p_corr, N; min_samples_per_class=0, allow_duplicates=false)
        @test length(indices_corr) == N
        @test length(unique(indices_corr)) == N
        @test DeconvUtil.fit_pdf(y[indices_corr], 1:n_classes) ≈ p_corr

        # training test split version with the same tests
        p_trn = rand(rng, Dirichlet(ones(n_classes)))
        p_trn_min = (round.(Int, N * p_trn) .- 1) / N
        p_trn_max = (round.(Int, N * p_trn) .+ 1) / N
        i_trn, i_tst = Data.subsample_indices(rng, y, p_trn, N, p, N; min_samples_per_class=(0,0), allow_duplicates=false)
        @test length(i_trn) == N
        @test length(i_tst) == N
        @test length(unique(vcat(i_trn, i_tst))) == 2*N
        @test all(DeconvUtil.fit_pdf(y[i_trn], 1:n_classes) .>= p_trn_min)
        @test all(DeconvUtil.fit_pdf(y[i_trn], 1:n_classes) .<= p_trn_max)
        @test all(DeconvUtil.fit_pdf(y[i_tst], 1:n_classes) .>= p_min)
        @test all(DeconvUtil.fit_pdf(y[i_tst], 1:n_classes) .<= p_max)

        # exhaust one class
        to_exhaust = rand(rng, 1:n_classes)
        y[y .== to_exhaust] = rand(
            rng,
            setdiff(1:n_classes, to_exhaust),
            sum(y .== to_exhaust)
        ) # replace the class to_exhaust with random other classes
        @test_throws Data.ExhaustedClassException Data.subsample_indices(rng, y, p_corr, N; n_classes=n_classes, min_samples_per_class=0, allow_duplicates=false)
    end
end # testset

@testset "Data.subsample_indices(...; min_samples_per_class=3, allow_duplicates=true)" begin
    for seed in 1:20
        # generate a random task
        rng = MersenneTwister(seed)
        n_classes = rand(rng, 3:10)
        n_samples = n_classes * rand(rng, 1000:10000)
        N = rand(rng, 500:2000)
        y = rand(rng, 1:n_classes, n_samples)
        @test length(unique(y)) == n_classes # check correctness

        # test min_samples_per_class
        p = rand(rng, Dirichlet(ones(n_classes)))
        p[rand(1:n_classes)] = 0.0
        indices = Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=3, allow_duplicates=true)
        @test length(indices) == N
        @test length(unique(y[indices])) == n_classes
        @test indices == Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=3, allow_duplicates=true)
        @test all(DeconvUtil.fit_pdf(y[indices], 1:n_classes; normalize=false) .>= 3)

        # test allow_duplicates
        p = rand(rng, Dirichlet(ones(n_classes)))
        y[y .== 2] .= 1 # remove class 2
        y[1] = 2 # introduce class 2 with a single item
        indices = Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=3, allow_duplicates=true)
        @test length(indices) == N
        @test length(unique(y[indices])) == n_classes
        @test indices == Data.subsample_indices(MersenneTwister(seed), y, p, N; min_samples_per_class=3, allow_duplicates=true)
        @test all(DeconvUtil.fit_pdf(y[indices], 1:n_classes; normalize=false) .>= 3)
        @test sum(indices .== 1) >= 3 # all min_samples_per_class are assigned to the one index
    end
end # testset
;
