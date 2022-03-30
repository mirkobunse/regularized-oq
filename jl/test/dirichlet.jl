using CherenkovDeconvolution, DeconvExp, Distributions, ScikitLearn, StatsBase
using ScikitLearn.CrossValidation: cross_val_score
@sk_import tree: DecisionTreeClassifier

M = 100000 # number of samples
N = 1000 # number of items per sample

p_natural = DeconvUtil.normalizepdf([ 1, 2, 6, 2 ])
C = length(p_natural)

dirichlet = Dirichlet(p_natural * N)

# correction is important: a DecisionTreeClassifier can _otherwise_ find the gaps and
# thereby produce 99% CV accuracy
p_dirichlet = hcat(map(p -> begin
    N_p = round.(Int, N * p)
    while N != sum(N_p)
        N_remaining = N - sum(N_p)
        if N_remaining > 0 # are additional draws needed?
            N_p[StatsBase.sample(1:C, Weights(max.(p, 1/N)), N_remaining)] .+= 1
        elseif N_remaining < 0 # are less draws needed?
            N_p[StatsBase.sample(1:C, Weights(max.(p, 1/N)), -N_remaining)] .-= 1
        end
    end # rarely needs more than one iteration
    N_p / N
end, eachcol(rand(dirichlet, M)))...)'

p_draw = vcat([ DeconvUtil.fit_pdf(StatsBase.sample(1:C, Weights(p_natural), N))' for _ in 1:M ]...)

X = vcat(p_dirichlet, p_draw)
y = vcat(zeros(Int, size(p_dirichlet, 1)), ones(Int, size(p_draw, 1)))
@info "Difference between p_dirichlet and p_draw" mean(cross_val_score(DecisionTreeClassifier(), X, y; cv=10))
