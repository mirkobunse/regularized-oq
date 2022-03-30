"""
    amazon(configfile="conf/job/gen/amazon.yml")

Experiments of different quantification methods on Amazon customer review text data.
"""
function amazon(configfile::String="conf/job/gen/amazon.yml")
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
    @info "Starting $(length(val_batches)) validation batch(es) on $(nworkers()) worker(s)."
    val_df = vcat(pmap(
        val_batch -> catch_during_trial(_amazon_batch, val_batch),
        val_batches
    )...)

    # split the validation data into curvature levels ∈ [ 1, 2, ..., n_splits ]
    val_df[!, :curvature_level] = _curvature_level(val_df, c[:protocol][:n_splits])

    # validation output
    CSV.write(c[:valfile], val_df)
    @info "Validation results written to $(c[:valfile])"

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
        method = String[],
        curvature_level = Int64[], # reason why this method is evaluated
        selection_metric = Symbol[], # also a reason why this method is evaluated
        sample = Int64[],
        sample_curvature = Float64[], # actual curvature of the respective sample
        nmd = Float64[],
        rnod = Float64[]
    ) # store all results in this DataFrame
    Random.seed!(batch[:seed])
    C_curv = LinearAlgebra.diagm(
        -1 => fill(-1, 4),
        0 => fill(2, 5),
        1 => fill(-1, 4)
    )[2:4, :] # matrix for curvature computation

    trials, vectorizer = _amazon_prefitted_trials(batch) # prefit all methods
    @info "Batch $(batch[:batch]) starts evaluating $(batch[:M_tst]) samples"
    for i in 1:batch[:M_tst]
        if i % 10 == 0
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
        f_true = fit_pdf(y_tst, 0:4)

        # deconvolve, evaluate, and store the results of all trials in this batch
        for trial in trials
            Random.seed!(trial[:seed])
            f_est = DeconvUtil.normalizepdf(deconvolve(trial[:prefitted_method], X_tst))
            nmd = Util.nmd(f_est, f_true)
            rnod = Util.rnod(f_est, f_true)
            sample_curvature = sum((C_curv*f_true).^2)

            # this model might be evaluated for multiple reasons; store the results for each reason
            for (cl, sm) in zip(trial[:method][:curvature_level], trial[:method][:selection_metric])
                push!(df, [ trial[:method][:name], trial[:method][:method_id], cl, sm, i, sample_curvature, nmd, rnod ])
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
        trial[:seed] = MetaConfigurations.find(trial[:method], :random_state)[1]
        Random.seed!(trial[:seed])
        @info "Batch $(batch[:batch]) training $(i_trial)/$(n_trials): $(trial[:method][:name])"
        trial[:prefitted_method] = prefit(configure_method(trial[:method]), X_trn, y_trn)
    end
    return trials, vectorizer
end

function _amazon_trial_batches(c::Dict{Symbol, Any})
    c = deepcopy(c)

    # collect configurations of each method
    methods = Dict{String,Vector{Dict{Symbol, Any}}}()
    for method in pop!(c, :method)
        id = method[:method_id]
        if !haskey(methods, id)
            push!(methods, id => Dict{Symbol, Any}[])
        end
        push!(methods[id], method)
    end

    # initialize empty batches
    batches = [ deepcopy(c) for _ in 1:nworkers() ]
    for (i, batch) in enumerate(batches)
        batch[:batch] = i
        batch[:method] = Dict{Symbol,Any}[]
    end

    # round-robin assignment of methods to batches
    i_batch = 1
    for methods_of_id in values(methods)
        for method in methods_of_id
            push!(batches[i_batch][:method], method)
            i_batch = (i_batch % length(batches)) + 1
        end
    end
    batches = filter(b -> length(b[:method]) > 0, batches)

    n_trials = sum([ length(b[:method]) for b in batches ])
    @info "Assigned $(n_trials) trials to $(length(batches)) batch(es)"
    return batches
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
                groupby(val_df, [:name, :method]),
                selection_metric => DataFrames.mean => :avg_metric
            ), :method), # average NMD/RNOD per configuration
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
                    groupby(val_df, [:name, :method, :curvature_level]),
                    selection_metric => DataFrames.mean => :avg_metric
                ), [:method, :curvature_level]),
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

function plot_amazon_smoothness(
        data_path::String="/mnt/data/isti_confidential/Books/",
        output_path::String="res/metrics/amazon_smoothness.csv"
        )
    f_trn = DeconvUtil.fit_pdf(load_amazon_data(data_path * "/training_data.txt")[2], 0:4)
    C = LinearAlgebra.diagm(
        -1 => fill(-1, 4),
        0 => fill(2, 5),
        1 => fill(-1, 4)
    )[2:4, :] # matrix for curvature computation

    # compute NMDs and curvatures for all validation samples
    data_path *= "app"
    nmd = zeros(1000)
    curv = zeros(1000)
    for i_val in 1:1000
        f_val = DeconvUtil.fit_pdf(load_amazon_data(data_path * "/dev_samples/$(i_val-1).txt")[2], 0:4)
        nmd[i_val] = Util.nmd(f_trn, f_val)
        curv[i_val] = sum((C*f_val).^2)
    end

    # divide NMDs and curvatures into three buckets
    nmd_split = Statistics.quantile(unique(nmd), 1/3 .* collect(1:(3-1)))
    nmd_level = vec(sum(hcat([ nmd .> s for s in nmd_split ]...); dims=2)) .+ 1
    curv_split = Statistics.quantile(unique(curv), 1/3 .* collect(1:(3-1)))
    curv_level = vec(sum(hcat([ curv .> s for s in curv_split ]...); dims=2)) .+ 1

    # inspect the similarity of the buckets
    confusion = zeros(3, 3)
    for i in 1:3, j in 1:3
        confusion[i, j] = sum((nmd_level .== i) .& (curv_level .== j))
    end
    @info "Confusion between NMD-defined and curvature-defined levels" confusion

    df = DataFrame(nmd=nmd, curvature=curv, nmd_level=nmd_level, curvature_level=curv_level)
    CSV.write(output_path, df)
    @info "NMD and curvature values written to $(output_path)"
    return df
end

# split each line by spaces, parse into Float64s, and reshape into a matrix
parse_dense_vector(X_txt::Vector{String}) =
    vcat(map(x -> parse.(Float64, split(x, r"\s+"))', X_txt)...)
