module Conf

using ComfyCommons: ComfyLogging, ComfyGit
using MetaConfigurations, Random
using ..Util, ..Data, ..MoreMethods
using CherenkovDeconvolution: DSEA, IBU, RUN, PRUN, SVD
using CherenkovDeconvolution: ConstantStepsize, ExpDecayStepsize, MulDecayStepsize, DEFAULT_STEPSIZE, RunStepsize, OptimizedStepsize
using CherenkovDeconvolution: TreeBinning, KMeansBinning
using CherenkovDeconvolution: NoSmoothing, PolynomialSmoothing

DISCRETE_CONSTRUCTORS = Dict(
    "run" => RUN,
    "prun" => PRUN,
    "ibu" => IBU,
    "svd" => SVD
)

QUAPY_CONSTRUCTORS = Dict(
    "cc" => MoreMethods.ClassifyAndCount,
    "pcc" => MoreMethods.ProbabilisticClassifyAndCount,
    "acc" => MoreMethods.AdjustedClassifyAndCount,
    "pacc" => MoreMethods.ProbabilisticAdjustedClassifyAndCount,
    "emq" => MoreMethods.ExpectationMaximizationQuantifier
) # multi-class quantifiers from QuaPy

"""
    configure_method(c::Dict{Symbol, Any}; inspect::Function=(args...)->nothing, kwargs...)

Setting up an Deconvolution Method from CherenkovDeconvolution.jl based on a configuration file.
"""
function configure_method(c::Dict{Symbol, Any}; inspect::Function=(args...)->nothing, fit_learner::Bool=true, kwargs...)
    c = copy(c) # keep the input unchanged; CAUTION: this copy is only shallow
    kwargs = Dict(kwargs)
    c[:inspect] = inspect

    # initialize the stepsize, if configured
    if :stepsize in keys(c)
        c[:stepsize] = _configure_stepsize(c[:stepsize]) # overwrite with Stepsize object
    end 

    # initialize smoothing, if configured
    if :smoothing in keys(c)
        c[:smoothing] = _configure_smoothing(c[:smoothing])
    end

    # initialize classifiers, binnings, and methods
    method_id = c[:method_id]
    if method_id in ["dsea", "dsea+"]
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        return DSEA(clf; _method_arguments(c)...)
    elseif method_id in keys(DISCRETE_CONSTRUCTORS)
        binning = _configure_binning(c[:binning])
        return DISCRETE_CONSTRUCTORS[method_id](binning; _method_arguments(c)...)
    elseif method_id == "oqt"
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        return MoreMethods.OQT(clf; _method_arguments(c)...)
    elseif method_id == "arc"
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        return MoreMethods.ARC(clf; _method_arguments(c)...)
    elseif method_id in keys(QUAPY_CONSTRUCTORS)
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        return QUAPY_CONSTRUCTORS[method_id](clf; fit_learner=fit_learner, _method_arguments(c)...)
    elseif method_id == "semq"
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        return MoreMethods.SmoothEMQ(clf; fit_learner=fit_learner, _method_arguments(c)...)
    elseif method_id ∈ ["nacc", "npacc"]
        clf = get(kwargs, :clf, _configure_classifier(c[:classifier]))
        if !fit_learner
            throw(ArgumentError("fit_learner=false is not yet implemented for NACC/NPACC"))
        end
        if method_id == "nacc"
            return MoreMethods.NACC(clf; _method_arguments(c)...)
        else
            return MoreMethods.NPACC(clf; _method_arguments(c)...)
        end
    else
        throw(ArgumentError("Unknown method_id=$(method_id)"))
    end
end

function _method_arguments(c::Dict{Symbol,Any})
    method_keys = if c[:method_id] in ["dsea", "dsea+"]
        [:epsilon, :f_0, :fixweighting, :inspect, :K, :n_bins_y, :return_contributions, :smoothing, :stepsize, :inspect]
    elseif c[:method_id] == "svd"
        [:B, :effective_rank, :epsilon_C, :fit_ratios, :n_bins_y, :N]
    elseif c[:method_id] == "ibu"
        [:epsilon, :f_0, :fit_ratios, :inspect, :K, :n_bins_y, :smoothing, :stepsize]
    elseif c[:method_id] == "run"
        [:acceptance_correction, :ac_regularisation, :epsilon, :fit_ratios, :inspect, :K, :log_constant, :n_bins_y, :n_df ]
    elseif c[:method_id] == "prun"
        [:acceptance_correction, :ac_regularisation, :epsilon, :f_0, :fit_ratios, :inspect, :K, :log_constant, :n_bins_y, :tau]
    elseif c[:method_id] == "oqt"
        [:epsilon, :val_split]
    elseif c[:method_id] == "arc"
        [:with_binary_tree_regressor]
    elseif c[:method_id] in ["acc", "pacc"]
        [:val_split]
    elseif c[:method_id] in ["semq"]
        [:smoothing]
    elseif c[:method_id] in ["nacc", "npacc"]
        [:val_split, :criterion, :regularization, :tau]
    else
        Symbol[] # other methods (cc, pcc, emq) have no additional arguments
    end
    arg = Dict{Symbol,Any}()
    for k in keys(c)
        if k in method_keys
            push!(arg, k => c[k])
        end
    end
    return arg
end

_configure_classifier(c) =
    Util.classifier_from_config(haskey(c, :skconfig) ? c[:skconfig] : c)

_configure_binning(c::Dict{Symbol, Any}) =
    if c[:method_id] == "tree"
        J = c[:J]
        criterion = get(c, :criterion, "gini")
        seed = get(c, :seed, rand(UInt32))
        TreeBinning(J; criterion = criterion, seed = seed)
    elseif c[:method_id] == "kmeans"
        J = c[:J]
        seed = get(c, :seed, rand(UInt32))
        KMeansBinning(J; seed = seed)
    elseif c[:method_id] == "classifier"
        MoreMethods.ClassificationBinning(_configure_classifier(c[:classifier]))
    else
        throw(ArgumentError("Unknown binning $(c[:method_id])"))
    end

function _configure_stepsize(c::Dict{Symbol, Any})
    method = c[:method_id]
    if method == "constant"
        alpha = get(c, :alpha, 1.0)
        ConstantStepsize(alpha)
    elseif method == "decay_mul"
        eta = c[:eta]
        a   = get(c, :a, 1.0)
        MulDecayStepsize(eta, a)   
    elseif method == "decay_exp"
        eta = c[:eta]
        a   = get(c, :a, 1.0)
        ExpDecayStepsize(eta, a)
    elseif method == "run"
        binning = _configure_binning(c[:binning])
        decay   = get(c, :decay, false)
        tau     = get(c, :tau, 0.0)
        warn    = get(c, :warn, false)
        RunStepsize(binning; decay = decay, tau = tau, warn = warn)
    elseif method == "optimal"
        # must set dynamically 
        return c[:optimal_stepsize]
    else
        throw(ArgumentError("Unknown stepsize method_id=$(method_id)"))
    end
end

function _configure_smoothing(c::Dict{Symbol, Any})
    c = copy(c) # keep the original configuration intact
    method = pop!(c, :method_id)
    if method == "polynomial"
        order = pop!(c, :order)
        return PolynomialSmoothing(order; c...)
    elseif method == "none"
        return NoSmoothing()
    else
        throw(ArgumentError("Unknown smoothing method_id=$(method_id)"))
    end
end

"""
    dirichlet([metaconfig = "conf/job/meta/dirichlet.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
function dirichlet(metaconfig::String="conf/job/meta/dirichlet.yml")

    # read config and check for recent changes
    meta = parsefile(metaconfig; dicttype=Dict{Symbol,Any})
    githaschanges = ComfyGit.haschanges("src", metaconfig)
    if (githaschanges) @warn "Uncommited changes may affect configurations" end

    # expand configuration
    for job in expand(meta, :dataset)
        interpolate!(job, :configfile, dataset=job[:dataset])
        interpolate!(job, :outfile, dataset=job[:dataset])
        interpolate!(job, :valfile, dataset=job[:dataset])

        # expand classifiers and interpolate classifier names
        Random.seed!(job[:seed])
        classifiers = filter(clf -> begin # filtering: ignore LR on FACT
            job[:dataset] != "fact" || clf[:classifier] != "LogisticRegression"
        end, vcat(map(clf -> begin # expansion
            on = if clf[:classifier] == "LogisticRegression"
                [[:parameters, :class_weight], [:parameters, :C]]
            elseif clf[:classifier] == "DecisionTreeClassifier"
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion]]
            end
            expand(clf, on...)
        end, pop!(job, :classifier))...))
        for clf in classifiers # interpolate classifier names
            clf[:parameters][:random_state] = rand(UInt32)
            if clf[:classifier] == "LogisticRegression"
                clf[:name] = replace(clf[:name], "\$(C)" => clf[:parameters][:C])
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            elseif clf[:classifier] == "DecisionTreeClassifier"
                clf[:name] = replace(clf[:name], "\$(max_depth)" => clf[:parameters][:max_depth])
                if clf[:parameters][:criterion] == "entropy"
                    clf[:name] = replace(clf[:name], "\$(criterion)" => "E")
                else # if it's not entropy, it's gini
                    clf[:name] = replace(clf[:name], "\$(criterion)" => "G")
                end
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else # if it's not uniform, its natural
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            else
                throw(ArgumentError("Unknown classifier $(clf[:classifier])"))
            end
        end

        # expand and filter experiments
        dim_f = length(Data.bins(Data.dataset(job[:dataset], readdata=false))) # dimension of f
        J_seed = Dict(map(
            J -> J => rand(UInt32),
            unique(vcat(MetaConfigurations.find(job[:method], :J)...))
        )...)
        job[:method] = filter(exp -> begin # filtering
            if exp[:method_id] == "run" && exp[:discretization][:method_id] == "expand"
                exp[:discretization][:factor] * dim_f <= exp[:binning][:J]
            elseif exp[:method_id] == "run" && exp[:discretization][:method_id] == "reduce"
                exp[:discretization][:factor] != 1
            elseif exp[:method_id] == "svd"
                exp[:effective_rank] <= dim_f
            else true end
        end, vcat(map(exp -> begin # expansion
            if exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "emq"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc", "acc", "pacc"]
                exp[:classifier] = classifiers
                expand(exp, :val_split, :classifier)
            elseif exp[:method_id] == "ibu"
                expand(exp,
                    [:smoothing, :order],
                    [:smoothing, :impact],
                    [:binning, :J]
                )
            elseif exp[:method_id] == "run"
                expand(exp,
                    [:discretization, :method_id],
                    [:discretization, :factor],
                    [:binning, :J]
                )
            elseif exp[:method_id] == "prun"
                expand(exp, :tau, [:binning, :J])
            elseif exp[:method_id] == "svd"
                expand(exp, :effective_rank, [:binning, :J])
            elseif exp[:method_id] == "semq"
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:smoothing, :order],
                    [:smoothing, :impact],
                )
            elseif exp[:method_id] ∈ ["nacc", "npacc"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, :val_split, :criterion, :regularization, :tau)
            else # other methods are not supported
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...))

        # interpolate method names (and seed the binnings of IBU & RUN)
        for exp in job[:method]
            name = exp[:name]
            if exp[:method_id] == "ibu"
                exp[:binning][:seed] = J_seed[exp[:binning][:J]]
                name = replace(name, "\$(J)" => exp[:binning][:J])
                name = replace(name, "\$(order)" => exp[:smoothing][:order])
                name = replace(name, "\$(impact)" => exp[:smoothing][:impact])
            elseif exp[:method_id] == "run"
                exp[:binning][:seed] = J_seed[exp[:binning][:J]]
                name = replace(name, "\$(J)" => exp[:binning][:J])
                if exp[:discretization][:factor] == 1
                    name = replace(name, "\$(method_id) with factor \$(factor)" => "no regularization")
                end
                name = replace(name, "\$(method_id)" => exp[:discretization][:method_id])
                name = replace(name, "\$(factor)" => exp[:discretization][:factor])
            elseif exp[:method_id] == "prun"
                exp[:binning][:seed] = J_seed[exp[:binning][:J]]
                name = replace(name, "\$(J)" => exp[:binning][:J])
                name = replace(name, "\$(tau)" => exp[:tau])
            elseif exp[:method_id] == "svd"
                exp[:binning][:seed] = J_seed[exp[:binning][:J]]
                name = replace(name, "\$(J)" => exp[:binning][:J])
                name = replace(name, "\$(effective_rank)" => exp[:effective_rank])
            elseif exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "emq", "oqt", "arc", "acc", "pacc", "semq", "nacc", "npacc"]
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
                if exp[:method_id] in ["oqt", "arc", "acc", "pacc", "nacc", "npacc"]
                    name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:val_split]))}")
                    if exp[:method_id] ∈ ["nacc", "npacc"]
                        name = replace(name, "\$(val_split)" => exp[:val_split])
                        name = replace(name, "\$(criterion)" => Dict("mse"=>"L_2", "mae"=>"L_1")[exp[:criterion]])
                        name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:regularization]])
                        name = replace(name, "\$(tau)" => exp[:tau])
                    end
                elseif exp[:method_id] == "semq"
                    name = replace(name, "\$(order)" => exp[:smoothing][:order])
                    name = replace(name, "\$(impact)" => exp[:smoothing][:impact])
                end
            end
            exp[:name] = name # replace with interpolation
        end

        # inspect the number of configurations for each method
        n_configurations = Dict{String,Int}()
        for exp in job[:method]
            id = exp[:method_id]
            n_configurations[id] = 1 + get(n_configurations, id, 0)
        end
        for (id, n) in pairs(n_configurations)
            @info "$(id) will be optimized over $(n) configurations"
        end

        # write job to file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)
    end
end

"""
    amazon([metaconfig = "conf/job/meta/amazon.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
function amazon(metaconfig::String="conf/job/meta/amazon.yml")

    # read config and check for recent changes
    meta = parsefile(metaconfig; dicttype=Dict{Symbol,Any})
    githaschanges = ComfyGit.haschanges("src", metaconfig)
    if (githaschanges) @warn "Uncommited changes may affect configurations" end

    # expand configuration
    for job in expand(meta, :data)
        interpolate!(job, :configfile, id=job[:data][:id])
        interpolate!(job, :outfile, id=job[:data][:id])
        interpolate!(job, :valfile, id=job[:data][:id])

        # expand classifiers, fix random seeds, and interpolate classifier names
        Random.seed!(job[:seed])
        classifiers = filter(clf -> begin # filtering: ignore LR on FACT
            job[:data][:id] != "tfidf" || clf[:classifier] != "DecisionTreeClassifier"
        end, vcat(map(clf -> begin # expansion
            on = if clf[:classifier] == "LogisticRegression"
                [[:parameters, :class_weight], [:parameters, :C]]
            elseif clf[:classifier] == "DecisionTreeClassifier"
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion]]
            end
            expand(clf, on...)
        end, pop!(job, :classifier))...))
        for clf in classifiers # interpolate classifier names
            clf[:parameters][:random_state] = rand(UInt32)
            if clf[:classifier] == "LogisticRegression"
                clf[:name] = replace(clf[:name], "\$(C)" => clf[:parameters][:C])
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            elseif clf[:classifier] == "DecisionTreeClassifier"
                clf[:name] = replace(clf[:name], "\$(max_depth)" => clf[:parameters][:max_depth])
                if clf[:parameters][:criterion] == "entropy"
                    clf[:name] = replace(clf[:name], "\$(criterion)" => "E")
                else # if it's not entropy, it's gini
                    clf[:name] = replace(clf[:name], "\$(criterion)" => "G")
                end
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else # if it's not uniform, its natural
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            else
                throw(ArgumentError("Unknown classifier $(clf[:classifier])"))
            end
        end

        # expand and filter experiments
        dim_f = 5 # dimension of f
        job[:method] = filter(exp -> begin # filtering
            if exp[:method_id] == "run" && exp[:discretization][:method_id] == "expand"
                exp[:discretization][:factor] * dim_f <= exp[:binning][:J]
            elseif exp[:method_id] == "run" && exp[:discretization][:method_id] == "reduce"
                exp[:discretization][:factor] != 1
            elseif exp[:method_id] == "svd"
                exp[:effective_rank] <= dim_f
            else true end
        end, vcat(map(exp -> begin # expansion
            exp = deepcopy(exp)
            if exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "emq"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc", "acc", "pacc"]
                exp[:classifier] = classifiers
                expand(exp, :val_split, :classifier)
            elseif exp[:method_id] == "ibu"
                if exp[:binning][:method_id] == "classifier"
                    exp[:binning][:classifier] = classifiers
                    expand(exp,
                        [:smoothing, :order],
                        [:smoothing, :impact],
                        [:binning, :classifier]
                    )
                elseif exp[:binning][:method_id] == "tree"
                    expand(exp,
                        [:smoothing, :order],
                        [:smoothing, :impact],
                        [:binning, :J]
                    )
                else
                    throw(ArgumentError("Binning method $(exp[:binning][:method_id]) is not known"))
                end
            elseif exp[:method_id] == "run"
                if exp[:binning][:method_id] == "classifier"
                    exp[:binning][:classifier] = classifiers
                    expand(exp,
                        [:discretization, :method_id],
                        [:discretization, :factor],
                        [:binning, :classifier]
                    )
                elseif exp[:binning][:method_id] == "tree"
                    expand(exp,
                        [:discretization, :method_id],
                        [:discretization, :factor],
                        [:binning, :J]
                    )
                else
                    throw(ArgumentError("Binning method $(exp[:binning][:method_id]) is not known"))
                end
            elseif exp[:method_id] == "prun"
                if exp[:binning][:method_id] == "classifier"
                    exp[:binning][:classifier] = classifiers
                    expand(exp, :tau, [:binning, :classifier])
                elseif exp[:binning][:method_id] == "tree"
                    expand(exp, :tau, [:binning, :J])
                else
                    throw(ArgumentError("Binning method $(exp[:binning][:method_id]) is not known"))
                end
            elseif exp[:method_id] == "svd"
                if exp[:binning][:method_id] == "classifier"
                    exp[:binning][:classifier] = classifiers
                    expand(exp, :effective_rank, [:binning, :classifier])
                elseif exp[:binning][:method_id] == "tree"
                    expand(exp, :effective_rank, [:binning, :J])
                else
                    throw(ArgumentError("Binning method $(exp[:binning][:method_id]) is not known"))
                end
            elseif exp[:method_id] == "semq"
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:smoothing, :order],
                    [:smoothing, :impact]
                )
            elseif exp[:method_id] ∈ ["nacc", "npacc"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, :val_split, :criterion, :regularization, :tau)
            else # other methods are not supported
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...))

        # interpolate method names
        for exp in job[:method]
            name = exp[:name]
            if haskey(exp, :classifier) && isa(exp[:classifier], Dict)
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
            elseif haskey(exp[:binning], :classifier)
                name = replace(name, "\$(classifier)" => exp[:binning][:classifier][:name])
            end
            if exp[:method_id] == "ibu" # IBU interpolates $(J)
                if haskey(exp[:binning], :J)
                    name = replace(name, "\$(J)" => exp[:binning][:J])
                end
                name = replace(name, "\$(order)" => exp[:smoothing][:order])
                name = replace(name, "\$(impact)" => exp[:smoothing][:impact])
            elseif exp[:method_id] == "run"
                if haskey(exp[:binning], :J)
                    name = replace(name, "\$(J)" => exp[:binning][:J])
                end
                if exp[:discretization][:factor] == 1
                    name = replace(name, "\$(method_id) with factor \$(factor)" => "no regularization")
                end
                name = replace(name, "\$(method_id)" => exp[:discretization][:method_id])
                name = replace(name, "\$(factor)" => exp[:discretization][:factor])
            elseif exp[:method_id] == "prun"
                if haskey(exp[:binning], :J)
                    name = replace(name, "\$(J)" => exp[:binning][:J])
                end
                name = replace(name, "\$(tau)" => exp[:tau])
            elseif exp[:method_id] == "svd"
                if haskey(exp[:binning], :J)
                    name = replace(name, "\$(J)" => exp[:binning][:J])
                end
                name = replace(name, "\$(effective_rank)" => exp[:effective_rank])
            elseif exp[:method_id] in ["oqt", "arc", "acc", "pacc", "nacc", "npacc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:val_split]))}")
                if exp[:method_id] ∈ ["nacc", "npacc"]
                    name = replace(name, "\$(val_split)" => exp[:val_split])
                    name = replace(name, "\$(criterion)" => Dict("mse"=>"L_2", "mae"=>"L_1")[exp[:criterion]])
                    name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:regularization]])
                    name = replace(name, "\$(tau)" => exp[:tau])
                end
            elseif exp[:method_id] == "semq"
                name = replace(name, "\$(order)" => exp[:smoothing][:order])
                name = replace(name, "\$(impact)" => exp[:smoothing][:impact])
            end
            exp[:name] = name # replace with interpolation
        end

        # inspect the number of configurations for each method
        n_configurations = Dict{String,Int}()
        for exp in job[:method]
            id = exp[:method_id]
            n_configurations[id] = 1 + get(n_configurations, id, 0)
        end
        for (id, n) in pairs(n_configurations)
            @info "$(id) will be optimized over $(n) configurations"
        end

        # write job to file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)
    end
end

end # module
