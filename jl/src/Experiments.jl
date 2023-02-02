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
    Random,
    Statistics,
    StatsBase
using ..Util, ..Data, ..Configuration

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

    val_c = deepcopy(c)
    val_c[:data][:tst_path] = joinpath(
        c[:data][:path],
        c[:protocol][:sampling],
        "dev_samples",
    ) # pretend we were testing, but use the development samples
    val_c[:M_tst] = val_c[:M_val]
    for m in val_c[:method]
        m[:curvature_level] = [ -1 ]
        m[:selection_metric] = [ :none ]
    end # fake reason for evaluating each model: -1 means to validate on all data
    val_batches = _amazon_trial_batches(val_c)
    val_df = DataFrame()
    if validate
        @info "Starting $(length(val_batches)) validation batches on $(nworkers()) worker(s)."
        val_df = vcat(pmap(
            val_batch -> catch_during_trial(_amazon_batch, val_batch),
            val_batches
        )...)

        # split the validation data into curvature levels ∈ [ 1, 2, ..., n_splits ]
        val_df[!, :curvature_level] = _curvature_level(val_df, c[:protocol][:n_splits])

        # validation output
        CSV.write(c[:valfile], val_df)
        @info "Validation results written to $(c[:valfile])"
    else
        val_df = coalesce.(CSV.read(c[:valfile], DataFrame), "")
        @info "Read validation results from $(c[:valfile])"
    end

    # select the best overall methods and the best methods for each curvature level
    _filter_best_methods!(c, val_df, c[:protocol][:n_splits])

    # parallel execution
    c[:data][:tst_path] = joinpath(
        c[:data][:path],
        c[:protocol][:sampling],
        "test_samples",
    )
    tst_batches = _amazon_trial_batches(c)
    @info "Starting $(length(tst_batches)) testing batch(es) on $(nworkers()) worker(s)."
    tst_df = vcat(pmap(
        tst_batch -> catch_during_trial(_amazon_batch, tst_batch),
        tst_batches
    )...)

    # also split the test data into curvature levels ∈ [ 1, 2, ..., n_splits ]
    rename!(tst_df, Dict(:curvature_level => :val_curvature_level)) # remember the reason for testing this model
    tst_df[!, :tst_curvature_level] = _curvature_level(tst_df, c[:protocol][:n_splits])

    # testing output
    CSV.write(c[:outfile], tst_df)
    @info "Testing results written to $(c[:outfile])"
    return tst_df
end

function _amazon_batch(batch::Dict{Symbol, Any})
    df = DataFrame(
        name = String[],
        validation_group = String[],
        curvature_level = Int64[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
    Util.numpy_seterr(invalid="ignore") # do not warn when an OOB score divides by NaN
    Random.seed!(batch[:seed])
    C_curv = LinearAlgebra.diagm(
        -1 => fill(-1, 4),
        0 => fill(2, 5),
        1 => fill(-1, 4)
    )[2:4, :] # matrix for curvature computation

    trials, vectorizer = _amazon_prefitted_trials(batch) # prefit all methods
    @info "Batch $(batch[:batch]) starts evaluating $(batch[:M_tst]) samples"
    for i in 1:batch[:M_tst]
        if i % 25 == 0
            @info "Batch $(batch[:batch]) has evaluated $(i)/$(batch[:M_tst]) samples"
        end
        X_txt, y_tst = load_amazon_data(batch[:data][:tst_path] * "/$(i-1).txt")
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

            # this model might be evaluated for multiple reasons; store the results for each reason
            for (cl, sm) in zip(trial[:method][:curvature_level], trial[:method][:selection_metric])
                validation_group = get(trial[:method], :validation_group, trial[:method][:method_id])
                push!(df, [ trial[:method][:name], validation_group, cl, sm, i, sample_curvature, nmd, rnod ])
            end
        end
    end
    return df
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
    X_trn = X_full[i_trn, :]
    y_trn = y_full[i_trn]
    X_val = X_full[i_val, :]
    y_val = y_full[i_val]
    X_tst = X_full[i_tst, :]
    y_tst = y_full[i_tst]

    # generate seeds for validation and test samples
    c[:val_seed] = rand(UInt32, c[:M_val])
    c[:tst_seed] = rand(UInt32, c[:M_tst])

    # configure and execute the validation step
    val_c = deepcopy(c) # keep the original configuration intact
    val_c[:tst_seed] = val_c[:val_seed] # pretend we were testing
    val_c[:N_tst] = val_c[:N_val]
    for m in val_c[:method]
        m[:curvature_level] = [ -1 ]
        m[:selection_metric] = [ :none ]
    end # fake reason for evaluating each model: -1 means to validate on all data
    val_trials = expand(val_c, :method)
    for (i, val_trial) in enumerate(val_trials)
        val_trial[:trial] = i # add the trial number to each configuration
    end

    if validate
        @info "Starting $(length(val_trials)) validation trials on $(nworkers()) worker(s)."
        job_args = [ X_trn, y_trn, X_val, y_val, discr ]
        val_df = vcat(pmap(
            val_trial -> catch_during_trial(_dirichlet_trial, val_trial, job_args...),
            val_trials
        )...)

        # split the validation data into curvature levels, see src/job/amazon.jl
        val_df[!, :curvature_level] = _curvature_level(val_df, c[:protocol][:n_splits])

        # validation output
        CSV.write(c[:valfile], val_df)
        @info "Validation results written to $(c[:valfile])"
    else
        val_df = coalesce.(CSV.read(c[:valfile], DataFrame), "")
        @info "Read validation results from $(c[:valfile])"
    end

    # select the best overall methods and the best methods for each curvature level
    _filter_best_methods!(c, val_df, c[:protocol][:n_splits])

    # parallel execution
    tst_trials = expand(c, :method)
    for (i, tst_trial) in enumerate(tst_trials)
        tst_trial[:trial] = i # add the trial number to each configuration
    end
    @info "Starting $(length(tst_trials)) testing trials on $(nworkers()) worker(s)."
    job_args = [ X_trn, y_trn, X_tst, y_tst, discr ]
    tst_df = vcat(pmap(
        tst_trial -> catch_during_trial(_dirichlet_trial, tst_trial, job_args...),
        tst_trials
    )...)

    # also split the test data into curvature levels, see src/job/amazon.jl
    rename!(tst_df, Dict(:curvature_level => :val_curvature_level))
    tst_df[!, :tst_curvature_level] = _curvature_level(tst_df, c[:protocol][:n_splits])

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
        discr :: AbstractDiscretizer
        ) where {TN<:Number, TL<:Number}
    n_classes = length(Data.bins(discr))
    f_trn = DeconvUtil.fit_pdf(y_trn, Data.bins(discr))
    df = DataFrame(
        name = String[],
        validation_group = String[],
        curvature_level = Int64[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
    Util.numpy_seterr(invalid="ignore") # do not warn when an OOB score divides by NaN
    C_curv = LinearAlgebra.diagm(
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

    # different parametrizations of the Dirichlet distribution realize APP and NPP
    dirichlet_parameters = if trial[:protocol][:sampling] == "app"
        ones(n_classes)
    elseif trial[:protocol][:sampling] == "npp"
        DeconvUtil.fit_pdf(y_tst) * trial[:N_tst]
    else
        throw(ArgumentError("Protocol '$(trial[:protocol][:sampling])' not supported"))
    end
    dirichlet_distribution = Dirichlet(dirichlet_parameters)

    # evaluate
    method = prefit(Configuration.configure_method(trial[:method]), X_trn, y_trn)
    for sample_seed in trial[:tst_seed] # draw samples with random prevalences
        rng_sample = MersenneTwister(sample_seed)
        p_sample = rand(rng_sample, dirichlet_distribution)
        i_sample = Data.subsample_indices(rng_sample, y_tst, p_sample, trial[:N_tst])
        f_true = DeconvUtil.fit_pdf(y_tst[i_sample], Data.bins(discr))

        # deconvolve, evaluate, and store the results
        f_est = DeconvUtil.normalizepdf(deconvolve(method, X_tst[i_sample, :]))
        nmd = Util.nmd(f_est, f_true)
        rnod = Util.rnod(f_est, f_true)
        sample_curvature = sum((C_curv*f_true).^2)

        # this model might be evaluated for multiple reasons; store the results for each reason
        for (cl, sm) in zip(trial[:method][:curvature_level], trial[:method][:selection_metric])
            validation_group = get(trial[:method], :validation_group, trial[:method][:method_id])
            push!(df, [ trial[:method][:name], validation_group, cl, sm, sample_seed, sample_curvature, nmd, rnod ])
        end
    end
    return df
end

# split sample curvatures into discrete levels of equal probability
function _curvature_level(df::DataFrame, n_splits::Int)
    if n_splits > 1
        split_points = Statistics.quantile(
            unique(df[!, :sample_curvature]),
            1/n_splits .* collect(1:(n_splits-1))
        )
        @info "Computing $(n_splits) curvature levels" split_points
        return vec(sum(hcat([
            df[!, :sample_curvature] .> s for s in split_points
        ]...); dims=2)) .+ 1
    else
        return fill(-1, nrow(df))
    end
end

# remove all but the best methods (for each curvature level) from the configuration c
function _filter_best_methods!(c::Dict{Symbol,Any}, val_df::DataFrame, n_splits::Int)

    # find the methods which perform best over the entire APP, according to NMD and RNOD
    best_methods = vcat(map(selection_metric -> begin
        best_avg = combine( # find methods with the minimum average metric
            groupby(combine(
                groupby(val_df, [:name, :validation_group]),
                selection_metric => DataFrames.mean => :avg_metric
            ), :validation_group), # average NMD/RNOD per configuration
            sdf -> sdf[argmin(sdf[!, :avg_metric]), :]
        )[!, [:name]]
        best_avg[!, :selection_metric] .= selection_metric # reason for keeping these methods
        best_avg # "return value" of the map operation
    end, [:nmd, :rnod])...)
    best_methods[!, :curvature_level] .= -1 # these methods are selected for the full APP

    # do the same for separate curvature levels
    if n_splits > 1
        for selection_metric in [:nmd, :rnod]
            best_avg = combine(
                groupby(combine( # also split by curvature_level, this time
                    groupby(val_df, [:name, :validation_group, :curvature_level]),
                    selection_metric => DataFrames.mean => :avg_metric
                ), [:validation_group, :curvature_level]),
                sdf -> sdf[argmin(sdf[!, :avg_metric]), :]
            )[!, [:name, :curvature_level]]
            best_avg[!, :selection_metric] .= selection_metric
            best_methods = vcat(best_methods, best_avg) # append
        end
    end

    # remove all methods that are not among the best ones
    c[:method] = filter(m -> m[:name] ∈ best_methods[!, :name], c[:method])
    for m in c[:method] # store the reason for keeping each method
        m[:curvature_level] = best_methods[best_methods[!, :name].==m[:name], :curvature_level]
        m[:selection_metric] = best_methods[best_methods[!, :name].==m[:name], :selection_metric]
    end
    list_best = [(n=m[:name], c=m[:curvature_level], m=m[:selection_metric]) for m in c[:method]]
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

    # different parametrizations of the Dirichlet distribution realize APP and NPP
    dirichlet_parameters = if c[:protocol][:sampling] == "app"
        ones(n_classes)
    elseif c[:protocol][:sampling] == "npp"
        DeconvUtil.fit_pdf(y_tst) * c[:N_tst]
    else
        throw(ArgumentError("Protocol '$(c[:protocol][:sampling])' not supported"))
    end
    dirichlet_distribution = Dirichlet(dirichlet_parameters)

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

    # filter smoothest 20% for APP-OQ
    val_indices = val_indices[1 .== _curvature_level(DataFrame(sample_curvature=val_curvatures), c[:protocol][:n_splits]), :]
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

    # filter smoothest 20% for APP-OQ
    tst_indices = tst_indices[1 .== _curvature_level(DataFrame(sample_curvature=tst_curvatures), c[:protocol][:n_splits]), :]
    CSV.write("app-oq_tst_indices.csv", DataFrame(tst_indices, :auto); writeheader=false)
    @info "Validation samples have been written to app-oq_tst_indices.csv"

    return nothing
end

end # module
