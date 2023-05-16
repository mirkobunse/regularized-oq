module Experiments

using
    CherenkovDeconvolution,
    CSV,
    DataFrames,
    Discretizers,
    Distributed,
    Distributions,
    LinearAlgebra,
    MetaConfigurations,
    PyCall,
    QUnfold,
    Random,
    Statistics,
    StatsBase
using ..Util, ..Data, ..Configuration
import Conda, Dates

const __castano_main = PyNULL()
const __castano_wrapper = PyNULL()
function __init__()
    castano_main = pyimport_e("ordinal_quantification.experiments.__main__")
    if ispynull(castano_main) # need to install ordinal_quantification?
        Conda.pip_interop(true)
        Conda.pip("install", "git+https://github.com/mirkobunse/ordinal_quantification.git")
        castano_main = pyimport("ordinal_quantification.experiments.__main__")
    end
    copy!(__castano_main, castano_main)

    LabelEncoder = pyimport("sklearn.preprocessing").LabelEncoder
    copy!(__castano_wrapper,
        @pydef mutable struct CastanoWrapper
            function __init__(self, method)
                self.method = method
            end
            function fit(self, X, y)
                self.encoder = LabelEncoder()
                y = self.encoder.fit_transform(y) .+ 1
                self.method = CherenkovDeconvolution.prefit(self.method, X, y)
                return self
            end
            function predict(self, X)
                try
                    pred = CherenkovDeconvolution.deconvolve(self.method, X)
                    if !all(isfinite.(pred))
                        @error "Prediction contains NaNs or Infs" worker=myid() method=self.method
                        C = length(self.encoder.classes_)
                        return ones(C) ./ C
                    else
                        return pred
                    end
                catch exception
                    showerror(stdout, exception, catch_backtrace())
                    println(stdout, "")
                    @error "An exception occured" worker=myid() method=self.method exception
                    C = length(self.encoder.classes_)
                    return ones(C) ./ C
                end
            end
        end
    )
end

"""
    run(configfile; kwargs...)

Hand the `configfile` to the Experiments method of which the name is configured by the property
`job` in the `configfile`. All `kwargs` are passed to this method.
"""
function run(configfile::String; kwargs...)
    c = parsefile(configfile)
    funname = "Experiments." * c["job"]
    @info "Calling $funname(\"$configfile\")" kwargs
    eval(Meta.parse(funname))(configfile; kwargs...) # function call
end

"""
    catch_during_trial(jobfunction, args...)

Call `jobfunction(args...)`, rethrowing any exception while printing the correct(!) stack
trace.
"""
catch_during_trial(jobfunction, args...) =
    try
        jobfunction(args...)
    catch exception
        showerror(stdout, exception, catch_backtrace())
        throw(exception)
    end

"""
    amazon(configfile="conf/gen/amazon.yml"; validate=true)

Experiments of different quantification methods on Amazon customer review text data. If `validate==false`, read the results from the `valfile` instead of running the validation / hyper-parameter optimization.
"""
function amazon(configfile::String="conf/gen/amazon.yml"; validate::Bool=true)
    c = parsefile(configfile; dicttype=Dict{Symbol,Any}) # read the configuration
    Random.seed!(c[:seed])
    @info "Parsing $(c[:data][:type]) from $(c[:data][:path])/"
    if haskey(c, :N_trn) # only used in testing configurations
        @warn "Limiting training data to $(c[:N_trn]) instances"
    end

    # configure the validation step: pretend we were testing
    val_c = deepcopy(c)
    val_c[:data][:tst_path] = joinpath(c[:data][:path], "app", "dev_samples")
    val_c[:data][:real_path] = joinpath(c[:data][:path], "real", "dev_samples")
    val_c[:M_tst] = val_c[:M_val]
    for m in val_c[:method] # fake reason for "testing" each model
        m[:protocol] = [ "none" ]
        m[:selection_metric] = [ :none ]
    end
    val_batches = _amazon_trial_batches(val_c)
    val_df = DataFrame()
    if validate
        @info "Starting $(length(val_batches)) validation batches on $(nworkers()) worker(s)."
        val_df = vcat(pmap(
            val_batch -> catch_during_trial(_amazon_batch, val_batch),
            val_batches
        )...)
        val_df[!, :protocol] = _protocol(val_df, c[:app_oq_frac])

        # validation output
        CSV.write(c[:valfile], val_df)
        @info "Validation results written to $(c[:valfile])"
    else
        val_df = coalesce.(CSV.read(c[:valfile], DataFrame), "")
        @info "Read validation results from $(c[:valfile])"
    end

    # select the best methods for each protocol
    _filter_best_methods!(c, val_df, c[:app_oq_frac])

    # parallel execution
    c[:data][:tst_path] = joinpath(c[:data][:path], "app", "test_samples")
    c[:data][:real_path] = joinpath(c[:data][:path], "real", "test_samples")
    tst_batches = _amazon_trial_batches(c)
    @info "Starting $(length(tst_batches)) testing batch(es) on $(nworkers()) worker(s)."
    tst_df = vcat(pmap(
        tst_batch -> catch_during_trial(_amazon_batch, tst_batch),
        tst_batches
    )...)
    rename!(tst_df, Dict(:protocol => :val_protocol))
    tst_df[!, :tst_protocol] = _protocol(tst_df, c[:app_oq_frac])

    # testing output
    CSV.write(c[:outfile], tst_df)
    @info "Testing results written to $(c[:outfile])"
    return tst_df
end

function _amazon_batch(batch::Dict{Symbol, Any})
    df = DataFrame(
        name = String[],
        validation_group = String[],
        protocol = String[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        is_real_sample = Bool[], # whether this sample is a real sample
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
    Util.numpy_seterr(invalid="ignore") # do not warn when an OOB score divides by NaN
    Random.seed!(batch[:seed])
    C = LinearAlgebra.diagm(
        -1 => fill(-1, 4),
        0 => fill(2, 5),
        1 => fill(-1, 4)
    )[2:4, :] # matrix for curvature computation

    trials, v = _amazon_prefitted_trials(batch) # prefit all methods
    @info "Batch $(batch[:batch]) starts evaluating $(batch[:M_tst]) samples"
    for i in 1:batch[:M_tst]
        if i % 25 == 0
            @info "Batch $(batch[:batch]) has evaluated $(i)/$(batch[:M_tst]) samples"
        end
        _amazon_evaluate!(df, batch, C, i, false, batch[:data][:tst_path], trials, v)
        if get(batch[:data], :evaluate_real_data, false)
            _amazon_evaluate!(df, batch, C, i, true, batch[:data][:real_path], trials, v)
        end
    end
    return df
end

function _amazon_evaluate!(
        df::AbstractDataFrame,
        batch::Dict{Symbol, Any},
        C_curv::Matrix{T},
        i::Int64,
        is_real_sample::Bool,
        tst_path::AbstractString,
        trials::Vector{Dict{Symbol, Any}},
        vectorizer::Any,
        ) where {T <: Real}
    X_txt, y_tst = load_amazon_data(tst_path * "/$(i-1).txt")
    X_tst = if batch[:data][:type] == "raw_text"
        vectorizer.transform(X_txt)
    elseif batch[:data][:type] == "dense_vector"
        parse_dense_vector(X_txt)
    else
        throw(ArgumentError("Data type $(batch[:data][:type]) is not known"))
    end # might be X_val during validation
    f_true = DeconvUtil.fit_pdf(y_tst, 0:4)

    # deconvolve, evaluate, and store the results of all trials in this batch
    for trial in trials
        Random.seed!(trial[:seed])
        f_est = DeconvUtil.normalizepdf(deconvolve(trial[:prefitted_method], X_tst))
        nmd = Util.nmd(f_est, f_true)
        rnod = Util.rnod(f_est, f_true)
        sample_curvature = sum((C_curv*f_true).^2)

        # a model can be evaluated for multiple reasons; store the results for each
        for (protocol, sm) in zip(
                trial[:method][:protocol],
                trial[:method][:selection_metric]
                )
            validation_group = get(
                trial[:method],
                :validation_group,
                trial[:method][:method_id]
            )
            push!(df, [ trial[:method][:name], validation_group, protocol, sm, i, sample_curvature, is_real_sample, nmd, rnod ])
        end
    end
end

function _amazon_prefitted_trials(batch::Dict{Symbol, Any})
    X_txt, y_trn = load_amazon_data(batch[:data][:path] * "/training_data.txt")
    if haskey(batch, :N_trn) # only used in testing configurations
        X_txt = X_txt[1:batch[:N_trn]]
        y_trn = y_trn[1:batch[:N_trn]]
    end
    vectorizer = Util.SkObject(
        "sklearn.feature_extraction.text.TfidfVectorizer";
        min_df = 5,
        sublinear_tf = true,
        ngram_range = (1,2)
    ) # the suggested TF-IDF representation; only used for raw_text data
    X_trn = if batch[:data][:type] == "raw_text"
        vectorizer.fit_transform(X_txt)
    elseif batch[:data][:type] == "dense_vector"
        parse_dense_vector(X_txt)
    else
        throw(ArgumentError("Data type $(batch[:data][:type]) is not known"))
    end

    trials = expand(batch, :method)
    n_trials = length(trials)
    for (i_trial, trial) in enumerate(trials)
        try
            trial[:seed] = MetaConfigurations.find(trial[:method], :random_state)[1]
        catch
            @error "no :random_state" trial[:method]
            rethrow()
        end
        Random.seed!(trial[:seed])
        if haskey(trial[:method], :classifier) && haskey(trial[:method][:classifier], :bagging)
            trial[:method][:classifier][:bagging][:n_estimators] = 3
        end
        @info "Batch $(batch[:batch]) training $(i_trial)/$(n_trials): $(trial[:method][:name])"
        trial[:prefitted_method] = prefit(Configuration.configure_method(trial[:method]), X_trn, y_trn)
    end
    return trials, vectorizer
end

function _amazon_trial_batches(c::Dict{Symbol, Any})
    c = deepcopy(c)

    # collect configurations of each method
    methods = Dict{String,Vector{Dict{Symbol, Any}}}()
    for method in pop!(c, :method)
        validation_group = get(method, :validation_group, method[:method_id])
        if !haskey(methods, validation_group)
            push!(methods, validation_group => Dict{Symbol, Any}[])
        end
        push!(methods[validation_group], method)
    end

    # initialize empty batches
    batches = [ deepcopy(c) for _ in 1:nworkers() ]
    for (i, batch) in enumerate(batches)
        batch[:batch] = i
        batch[:method] = Dict{Symbol,Any}[]
    end

    # round-robin assignment of methods to batches
    i_batch = 1
    for methods_of_group in values(methods)
        for method in methods_of_group
            push!(batches[i_batch][:method], method)
            i_batch = (i_batch % length(batches)) + 1
        end
    end
    batches = filter(b -> length(b[:method]) > 0, batches)

    n_trials = sum([ length(b[:method]) for b in batches ])
    @info "Assigned $(n_trials) trials to $(length(batches)) batch(es)"
    return batches
end

# To-Do: move to Data module?
function load_amazon_data(data_path::String)
    X = String[] # raw text samples
    y = Int[] # labels on an ordinal scale
    open(data_path) do textfile
        for line in eachline(textfile)
            push!(y, parse(Int, line[1])) # assume label on first position
            push!(X, line[3:end]) # extract text
        end
    end
    return X, y
end

# split each line by spaces, parse into Float64s, and reshape into a matrix
parse_dense_vector(X_txt::Vector{String}) =
    vcat(map(x -> parse.(Float64, split(x, r"\s+"))', X_txt)...)

# read amazon prevalences
function load_amazon_prevalences(
        prevalence_path::String = "data/prevalence_votes1_reviews100.csv";
        shuffle::Bool = true
        )
    p = Matrix{Float64}(CSV.read(prevalence_path, DataFrame; header=false))
    if !shuffle
        return p
    end
    return hcat(shuffle(collect(eachrow(p)))...)' # shuffle rows
end

"""
    dirichlet(configfile="conf/gen/dirichlet_fact.yml"; validate=true)

Comparative evaluation over the unit simplex. If `validate==false`, read the results from the `valfile` instead of running the validation / hyper-parameter optimization.
"""
function dirichlet(configfile::String="conf/gen/dirichlet_fact.yml"; validate::Bool=true)
    c = parsefile(configfile; dicttype=Dict{Symbol,Any}) # read the configuration

    # read and split the data
    Random.seed!(c[:seed])
    dataset = Data.dataset(c[:dataset])
    discr = Data.discretizer(dataset)
    X_full = Data.X_data(dataset)
    y_full = encode(discr, Data.y_data(dataset))
    df_acceptance = if c[:dataset] == "fact"
        load_acceptance()
    else
        nothing
    end
    i_trn, i_rem = Util.SkObject(
        "sklearn.model_selection.train_test_split",
        collect(1:length(y_full)); # split indices
        train_size = c[:N_trn],
        stratify = y_full,
        random_state = rand(UInt32)
    )
    i_val, i_tst = Util.SkObject(
        "sklearn.model_selection.train_test_split",
        i_rem;
        train_size = .5,
        stratify = y_full[i_rem],
        random_state = rand(UInt32)
    )
    @info "Split" length(y_full) length(i_trn) length(i_val) length(i_tst) length(unique(vcat(i_trn, i_val, i_tst))) df_acceptance!=nothing c[:dataset]
    X_trn = X_full[i_trn, :]
    y_trn = y_full[i_trn]
    X_val = X_full[i_val, :]
    y_val = y_full[i_val]
    X_tst = X_full[i_tst, :]
    y_tst = y_full[i_tst]

    # generate seeds for validation and test samples
    c[:val_seed] = rand(UInt32, c[:M_val])
    c[:tst_seed] = rand(UInt32, c[:M_tst])

    # configure the validation step: pretend we were testing
    val_c = deepcopy(c) # keep the original configuration intact
    val_c[:tst_seed] = val_c[:val_seed]
    val_c[:N_tst] = val_c[:N_val]
    for m in val_c[:method] # fake reason for "testing" each model
        m[:protocol] = [ "none" ]
        m[:selection_metric] = [ :none ]
    end
    val_trials = expand(val_c, :method)
    for (i, val_trial) in enumerate(val_trials)
        val_trial[:trial] = i # add the trial number to each configuration
    end

    if validate
        @info "Starting $(length(val_trials)) validation trials on $(nworkers()) worker(s)."
        job_args = [ X_trn, y_trn, X_val, y_val, discr, df_acceptance ]
        val_df = vcat(pmap(
            val_trial -> catch_during_trial(_dirichlet_trial, val_trial, job_args...),
            val_trials
        )...)
        val_df[!, :protocol] = _protocol(val_df, c[:app_oq_frac])

        # validation output
        CSV.write(c[:valfile], val_df)
        @info "Validation results written to $(c[:valfile])"
    else
        val_df = coalesce.(CSV.read(c[:valfile], DataFrame), "")
        @info "Read validation results from $(c[:valfile])"
    end

    # select the best methods for each protocol
    _filter_best_methods!(c, val_df, c[:app_oq_frac])

    # parallel execution
    tst_trials = expand(c, :method)
    for (i, tst_trial) in enumerate(tst_trials)
        tst_trial[:trial] = i # add the trial number to each configuration
    end
    @info "Starting $(length(tst_trials)) testing trials on $(nworkers()) worker(s)."
    job_args = [ X_trn, y_trn, X_tst, y_tst, discr, df_acceptance ]
    tst_df = vcat(pmap(
        tst_trial -> catch_during_trial(_dirichlet_trial, tst_trial, job_args...),
        tst_trials
    )...)
    rename!(tst_df, Dict(:protocol => :val_protocol))
    tst_df[!, :tst_protocol] = _protocol(tst_df, c[:app_oq_frac])

    # testing output
    CSV.write(c[:outfile], tst_df)
    @info "Testing results written to $(c[:outfile])"
    return tst_df
end

function _dirichlet_trial(
        trial :: Dict{Symbol, Any},
        X_trn :: Matrix{TN},
        y_trn :: Vector{TL},
        X_tst :: Matrix{TN}, # might be X_val during validation
        y_tst :: Vector{TL},
        discr :: AbstractDiscretizer,
        df_acceptance :: Union{DataFrame, Nothing}, # if something, evaluate on real data
        ) where {TN<:Number, TL<:Number}
    n_classes = length(Data.bins(discr))
    f_trn = DeconvUtil.fit_pdf(y_trn, Data.bins(discr))
    df = DataFrame(
        name = String[],
        validation_group = String[],
        protocol = String[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        is_real_sample = Bool[],
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
    Util.numpy_seterr(invalid="ignore") # do not warn when an OOB score divides by NaN
    C = LinearAlgebra.diagm(
        -1 => fill(-1, n_classes-1),
        0 => fill(2, n_classes),
        1 => fill(-1, n_classes-1)
    )[2:n_classes-1, :] # matrix for curvature computation
    seed = if haskey(trial[:method], :binning) && haskey(trial[:method][:binning], :seed)
        trial[:method][:binning][:seed] # binning seed
    else # find the random_state of the classifier
        MetaConfigurations.find(trial[:method], :random_state)[1]
    end
    Random.seed!(seed)
    @info "Trial $(trial[:trial]) is starting: $(trial[:method][:name])"

    # implement APP through a Dirichlet distribution
    dirichlet_distribution = Dirichlet(ones(n_classes))

    # evaluate on samples with random prevalences
    method = prefit(Configuration.configure_method(trial[:method]), X_trn, y_trn)
    for (i_seed, sample_seed) in enumerate(trial[:tst_seed])
        rng_sample = MersenneTwister(sample_seed)
        p_sample = rand(rng_sample, dirichlet_distribution)
        _dirichlet_evaluate!(df, trial, X_tst, y_tst, discr, C, method, sample_seed, rng_sample, p_sample, false)
        if df_acceptance != nothing
            p_sample = sample_poisson(rng_sample, trial[:N_tst], df_acceptance)
            _dirichlet_evaluate!(df, trial, X_tst, y_tst, discr, C, method, sample_seed, rng_sample, p_sample, true)
        end
        if i_seed % 50 == 0
            @info "Trial $(trial[:trial]) of $(trial[:method][:name]) has evaluated $(i)/$(trial[:M_tst]) samples"
        end
    end
    return df
end

function load_acceptance(path::String="data/fact_acceptance.csv")
    df_acceptance = CSV.read(path, DataFrame)
    return DataFrame(
        a_eff = df_acceptance[2:end-1, :a_eff],
        bin_center = df_acceptance[2:end-1, :bin_center]
    )
end

function sample_poisson(
        rng :: AbstractRNG = Random.GLOBAL_RNG,
        N :: Integer = 1000,
        df_acceptance :: DataFrame = load_acceptance();
        round :: Bool = true
        )
    p = magic_crab_flux(df_acceptance[!, :bin_center]) .* df_acceptance[!, :a_eff]
    λ = p * N ./ sum(p) # Poisson rates for N events in total
    random_sample = [ rand(rng, Poisson(λ_i)) for λ_i in λ ]
    if round
        return round_Np(rng, N, random_sample ./ sum(random_sample)) ./ N
    else
        return random_sample ./ sum(random_sample)
    end
end

"""
    magic_crab_flux(x)

Compute the Crab nebula flux in `GeV⋅cm²⋅s` for a vector `x` of energy values
that are given in `GeV`. This parametrization is by Aleksić et al. (2015).
"""
magic_crab_flux(x::Union{Float64,Vector{Float64}}) =
    @. 3.23e-10 * (x/1e3)^(-2.47 - 0.24 * log10(x/1e3))

"""
    round_Np([rng, ]N, p; Np_min=1)

Round `N * p` such that `sum(N*p) == N` and `minimum(N*p) >= Np_min`. We use this
rounding to determine the number of samples to draw according to `N` and `p`.
"""
function round_Np(rng::AbstractRNG, N::Int, p::Vector{Float64}; Np_min::Int=1)
    Np = max.(round.(Int, N * p), Np_min)
    while N != sum(Np)
        ϵ = N - sum(Np)
        if ϵ > 0 # are additional draws needed?
            Np[StatsBase.sample(rng, 1:length(p), Weights(max.(p, 1/N)), ϵ)] .+= 1
        elseif ϵ < 0 # are less draws needed?
            c = findall(Np .> Np_min)
            Np[StatsBase.sample(rng, c, Weights(max.(p[c], 1/N)), -ϵ)] .-= 1
        end
    end # rarely needs more than one iteration
    return Np
end

function _dirichlet_evaluate!(
        df :: DataFrame,
        trial :: Dict{Symbol, Any},
        X_tst :: Matrix{TN}, # might be X_val during validation
        y_tst :: Vector{TL},
        discr :: AbstractDiscretizer,
        C_curv :: Matrix{T},
        method :: Any,
        sample_seed :: Integer,
        rng_sample :: AbstractRNG,
        p_sample :: Vector{Float64},
        is_real_sample :: Bool,
        ) where {T <: Real, TN<:Number, TL<:Number}
    i_sample = Data.subsample_indices(rng_sample, y_tst, p_sample, trial[:N_tst])
    f_true = DeconvUtil.fit_pdf(y_tst[i_sample], Data.bins(discr))

    # deconvolve, evaluate, and store the results
    f_est = DeconvUtil.normalizepdf(deconvolve(method, X_tst[i_sample, :]))
    nmd = Util.nmd(f_est, f_true)
    rnod = Util.rnod(f_est, f_true)
    sample_curvature = sum((C_curv*f_true).^2)

    # this model might be evaluated for multiple reasons; store the results for each reason
    for (protocol, sm) in zip(trial[:method][:protocol], trial[:method][:selection_metric])
        validation_group = get(trial[:method], :validation_group, trial[:method][:method_id])
        push!(df, [ trial[:method][:name], validation_group, protocol, sm, sample_seed, sample_curvature, is_real_sample, nmd, rnod ])
    end
end

# does a sample curvature belong to APP, APP-OQ, or to the real protocol?
function _protocol(df::DataFrame, app_oq_frac::Real)
    is_app = .!(df[!, :is_real_sample])
    split_point = max(
        Statistics.quantile(unique(df[is_app, :sample_curvature]), app_oq_frac),
        minimum(df[is_app, :sample_curvature])
    )
    protocol = fill("real", nrow(df))
    protocol[is_app] = [
        if x <= split_point
            "app-oq"
        else
            "app"
        end
        for x ∈ df[is_app, :sample_curvature]
    ]
    return protocol
end

# remove all but the best methods (for each protocol) from the configuration c
function _filter_best_methods!(c::Dict{Symbol,Any}, val_df::DataFrame, app_oq_frac::Real)
    best_methods = vcat(map( # find the best methods for APP and APP-OQ
        protocol -> begin # outer map operation
            best_app = vcat(map(
                selection_metric -> begin # inner map operation
                    best_avg = combine( # find methods with the minimum average metric
                        groupby(combine(
                            groupby(
                                if protocol == "real"
                                    val_df[val_df[!, :protocol] .== "real", :]
                                elseif protocol == "app-oq" # use a subset (APP-OQ)
                                    val_df[val_df[!, :protocol] .== "app-oq", :]
                                else # use all non-real data (APP)
                                    val_df[val_df[!, :protocol] .!= "real", :]
                                end,
                                [:name, :validation_group]
                            ),
                            selection_metric => DataFrames.mean => :avg_metric
                        ), :validation_group), # average NMD/RNOD per configuration
                        sdf -> sdf[argmin(sdf[!, :avg_metric]), :]
                    )[!, [:name]]
                    best_avg[!, :selection_metric] .= selection_metric
                    best_avg # "return value" of the inner map operation
                end,
                [:nmd, :rnod] # apply the inner map to both selection metrics
            )...)
            best_app[!, :protocol] .= protocol
            best_app # "return value" of the outer map operation
        end,
        [ "real", "app-oq", "app" ] # apply the outer map to each protocol
    )...)

    # remove all methods that are not among the best ones
    c[:method] = filter(m -> m[:name] ∈ best_methods[!, :name], c[:method])
    for m in c[:method] # store the reason for keeping each method
        m[:protocol] = best_methods[best_methods[!, :name].==m[:name], :protocol]
        m[:selection_metric] = best_methods[best_methods[!, :name].==m[:name], :selection_metric]
    end
    list_best = [(n=m[:name], p=m[:protocol], m=m[:selection_metric]) for m in c[:method]]
    @info "Methods to evaluate on the test data" list_best
    return c
end

"""
    dirichlet_indices(configfile="conf/gen/dirichlet_fact.yml")

Extract the indices of our evaluation.
"""
function dirichlet_indices(configfile::String="conf/gen/dirichlet_fact.yml")
    c = parsefile(configfile; dicttype=Dict{Symbol,Any}) # read the configuration

    # read and split the data
    Random.seed!(c[:seed])
    dataset = Data.dataset(c[:dataset])
    discr = Data.discretizer(dataset)
    n_classes = length(Data.bins(discr))
    y_full = encode(discr, Data.y_data(dataset))
    i_trn, i_rem = Util.SkObject(
        "sklearn.model_selection.train_test_split",
        collect(1:length(y_full)); # split indices
        train_size = c[:N_trn],
        stratify = y_full,
        random_state = rand(UInt32)
    )
    i_val, i_tst = Util.SkObject(
        "sklearn.model_selection.train_test_split",
        i_rem;
        train_size = .5,
        stratify = y_full[i_rem],
        random_state = rand(UInt32)
    )
    @info "Split" length(y_full) length(i_trn) length(i_val) length(i_tst) length(unique(vcat(i_trn, i_val, i_tst)))
    y_trn = y_full[i_trn]
    y_val = y_full[i_val]
    y_tst = y_full[i_tst]
    C_curv = LinearAlgebra.diagm(
        -1 => fill(-1, n_classes-1),
        0 => fill(2, n_classes),
        1 => fill(-1, n_classes-1)
    )[2:n_classes-1, :] # matrix for curvature computation

    # generate seeds for validation and test samples
    c[:val_seed] = rand(UInt32, c[:M_val])
    c[:tst_seed] = rand(UInt32, c[:M_tst])

    # implement APP through a Dirichlet distribution
    dirichlet_distribution = Dirichlet(ones(n_classes))

    # generate indices
    @info "Generating indices for $(c[:M_val]) validation samples."
    val_indices = zeros(Int, (c[:M_val], c[:N_val]))
    val_curvatures = zeros(c[:M_val])
    for (sample_index, sample_seed) in enumerate(c[:val_seed])
        rng_sample = MersenneTwister(sample_seed)
        p_sample = rand(rng_sample, dirichlet_distribution)
        i_sample = Data.subsample_indices(rng_sample, y_val, p_sample, c[:N_val])
        f_true = DeconvUtil.fit_pdf(y_val[i_sample], Data.bins(discr))
        val_indices[sample_index, :] = i_sample
        val_curvatures[sample_index] = sum((C_curv*f_true).^2)
    end
    CSV.write("app_val_indices.csv", DataFrame(val_indices, :auto); writeheader=false)
    @info "Validation samples have been written to app_val_indices.csv"

    # filter the smoothest app_oq_frac % of the samples for APP-OQ
    val_indices = val_indices[_protocol(DataFrame(sample_curvature=val_curvatures), c[:app_oq_frac]) .== "app-oq", :]
    CSV.write("app-oq_val_indices.csv", DataFrame(val_indices, :auto); writeheader=false)
    @info "Validation samples have been written to app-oq_val_indices.csv"

    @info "Generating indices for $(c[:M_tst]) testing samples."
    tst_indices = zeros(Int, (c[:M_tst], c[:N_tst]))
    tst_curvatures = zeros(c[:M_tst])
    for (sample_index, sample_seed) in enumerate(c[:tst_seed])
        rng_sample = MersenneTwister(sample_seed)
        p_sample = rand(rng_sample, dirichlet_distribution)
        i_sample = Data.subsample_indices(rng_sample, y_tst, p_sample, c[:N_tst])
        f_true = DeconvUtil.fit_pdf(y_tst[i_sample], Data.bins(discr))
        tst_indices[sample_index, :] = i_sample
        tst_curvatures[sample_index] = sum((C_curv*f_true).^2)
    end
    CSV.write("app_tst_indices.csv", DataFrame(tst_indices, :auto); writeheader=false)
    @info "Validation samples have been written to app_tst_indices.csv"

    # filter the smoothest app_oq_frac % of the samples for APP-OQ
    tst_indices = tst_indices[_protocol(DataFrame(sample_curvature=tst_curvatures), c[:app_oq_frac]) .== "app-oq", :]
    CSV.write("app-oq_tst_indices.csv", DataFrame(tst_indices, :auto); writeheader=false)
    @info "Validation samples have been written to app-oq_tst_indices.csv"

    return nothing
end

"""
    castano(configfile="conf/gen/castano.yml"; n_jobs=1, is_test_run=false)

Comparative evaluation in the setup of Castano et al. (2022).
"""
function castano(
        configfile::String="conf/gen/castano.yml";
        n_jobs::Int = 1,
        is_test_run::Bool = false,
        )
    c = parsefile(configfile; dicttype=Dict{Symbol,Any}) # read the configuration
    methods = Array{Any}(c[:castano_methods])
    method_names = copy(c[:castano_methods])
    for method_config in c[:method] # extend methods with Julia callables
        push!(methods, (clf, _) -> _castano_method(method_config, clf))
        push!(method_names, method_config[:name])
    end

    # same configuration as in the original experiment
    n_bags = 300
    n_reps = 10
    n_folds = 20
    estimator_grid = Dict{String,Any}([
        "n_estimators" => [100],
        "max_depth" => [1, 5, 10, 15, 20, 25, 30],
        "min_samples_leaf" => [1, 2, 5, 10, 20],
    ])
    dataset_names = [
        "SWD",
        "ESL",
        "LEV",
        "cement_strength_gago",
        "stock.ord",
        "auto.data.ord_chu",
        "bostonhousing.ord_chu",
        "californiahousing_gago",
        "winequality-red_gago",
        "winequality-white_gago_rev",
        "skill_gago",
        "SkillCraft1_rev_7clases",
        "kinematics_gago",
        "SkillCraft1_rev_8clases",
        "ERA",
        "ailerons_gago",
        "abalone.ord_chu",
    ]
    if is_test_run
        @warn "This is a test run; results are not meaningful"
        estimator_grid = Dict{String,Any}([
            "n_estimators" => [10],
            "max_depth" => [1, 5],
            "min_samples_leaf" => [1],
        ])
        n_bags = 10
        n_reps = 2
        n_folds = 2
        dataset_names = [ "ESL" ]
    end
    output_dir = "res/castano/" * Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    mkpath(output_dir)
    config = Dict{String,Any}([
        "seed" => c[:seed],
        "n_bags" => n_bags,
        "n_reps" => n_reps,
        "n_folds" => n_folds,
        "option" => "CV(DECOMP)",
        "decomposer" => "Monotone",
        "n_jobs" => n_jobs,
        "output_dir" => output_dir,
        "estimator" => pyimport("sklearn.ensemble").RandomForestClassifier(
            random_state = c[:seed],
            class_weight = "balanced",
            oob_score = true,
        ),
        "estimator_grid" => estimator_grid,
        "methods" => methods,
    ])

    # parallel execution of each trial in Python
    @sync @distributed for (i, d) in collect(Iterators.product(1:n_reps, dataset_names))
        @info "Starting repetition $i of dataset $d on worker $(myid())"
        __castano_main._repetition_dataset(i, d, config)
    end

    # process the results
    config["methods"] = method_names
    df = CSV.read(__castano_main._collect_results(config), DataFrame)
    return df
end

_castano_method(c::Dict{Symbol,Any}, clf::Any) =
    PyObject(c[:name]) => __castano_wrapper(
        Configuration.configure_method(c, pyimport("sklearn.base").clone(clf))
    )

end # module
