"""
    dirichlet(configfile="conf/job/gen/dirichlet_fact.yml")

Comparative evaluation over the unit simplex.
"""
function dirichlet(configfile::String="conf/job/gen/dirichlet_fact.yml")
    c = parsefile(configfile; dicttype=Dict{Symbol,Any}) # read the configuration

    # read and split the data
    Random.seed!(c[:seed])
    dataset = Data.dataset(c[:dataset])
    discr = Data.discretizer(dataset)
    X_full = Data.X_data(dataset)
    y_full = encode(discr, Data.y_data(dataset))
    i_trn, i_rem = pyimport("sklearn.model_selection").train_test_split(
        collect(1:length(y_full)); # split indices
        train_size = c[:N_trn],
        stratify = y_full,
        random_state = rand(UInt32)
    )
    i_val, i_tst = pyimport("sklearn.model_selection").train_test_split(
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
        method = String[],
        curvature_level = Int64[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
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
    method = prefit(configure_method(trial[:method]), X_trn, y_trn)
    for sample_seed in trial[:tst_seed] # draw samples with random prevalences
        rng_sample = MersenneTwister(sample_seed)
        p_sample = rand(rng_sample, dirichlet_distribution)
        i_sample = Data.subsample_indices(rng_sample, y_tst, p_sample, trial[:N_tst])
        f_true = fit_pdf(y_tst[i_sample], Data.bins(discr))

        # deconvolve, evaluate, and store the results
        f_est = DeconvUtil.normalizepdf(deconvolve(method, X_tst[i_sample, :]))
        nmd = Util.nmd(f_est, f_true)
        rnod = Util.rnod(f_est, f_true)
        sample_curvature = sum((C_curv*f_true).^2)

        # this model might be evaluated for multiple reasons; store the results for each reason
        for (cl, sm) in zip(trial[:method][:curvature_level], trial[:method][:selection_metric])
            push!(df, [ trial[:method][:name], trial[:method][:method_id], cl, sm, sample_seed, sample_curvature, nmd, rnod ])
        end
    end
    return df
end
