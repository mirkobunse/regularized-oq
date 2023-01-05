module Configuration

using CherenkovDeconvolution, MetaConfigurations, Random
using ..Util, ..Data, ..MoreMethods

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
    "sld" => MoreMethods.ExpectationMaximizationQuantifier
) # multi-class quantifiers from QuaPy

"""
    configure_method(c::Dict{Symbol, Any})

Set up an OQ method from CherenkovDeconvolution.jl or from src/MoreMethods.jl.
"""
function configure_method(c::Dict{Symbol, Any})
    c = copy(c) # keep the input unchanged; CAUTION: this copy is only shallow

    # initialize smoothings and stepsizes, if configured
    if :smoothing in keys(c) # overwrite configuration with a Smoothing object?
        c[:smoothing] = _configure_smoothing(c[:smoothing])
    end
    if :stepsize in keys(c) # overwrite configuration with a Stepsize object?
        c[:stepsize] = _configure_stepsize(c[:stepsize])
    end

    # initialize classifiers, binnings, and methods
    kwargs = _method_arguments(c)
    if c[:method_id] in ["dsea", "dsea+"]
        return DSEA(_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] in keys(DISCRETE_CONSTRUCTORS) # RUN, IBU, ...
        return DISCRETE_CONSTRUCTORS[c[:method_id]](_configure_binning(c[:binning]); kwargs...)
    elseif c[:method_id] == "oqt"
        return MoreMethods.OQT(_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] == "arc"
        return MoreMethods.ARC(_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] in keys(QUAPY_CONSTRUCTORS)
        return QUAPY_CONSTRUCTORS[c[:method_id]](_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] == "osld"
        return MoreMethods.OrdinalEMQ(_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] == "oacc"
        return MoreMethods.OrdinalACC(_configure_classifier(c[:classifier]); kwargs...)
    elseif c[:method_id] == "opacc"
        return MoreMethods.OrdinalPACC(_configure_classifier(c[:classifier]); kwargs...)
    else
        throw(ArgumentError("Unknown method_id=$(c[:method_id])"))
    end
end

function _method_arguments(c::Dict{Symbol,Any})
    method_keys = if c[:method_id] in ["dsea", "dsea+"]
        [:epsilon, :f_0, :fixweighting, :K, :n_bins_y, :return_contributions, :smoothing, :stepsize]
    elseif c[:method_id] == "svd"
        [:B, :effective_rank, :epsilon_C, :fit_ratios, :n_bins_y, :N]
    elseif c[:method_id] == "ibu"
        [:epsilon, :f_0, :fit_ratios, :K, :n_bins_y, :smoothing, :stepsize]
    elseif c[:method_id] == "run"
        [:acceptance_correction, :ac_regularisation, :epsilon, :fit_ratios, :K, :log_constant, :n_bins_y, :n_df ]
    elseif c[:method_id] == "prun"
        [:acceptance_correction, :ac_regularisation, :epsilon, :f_0, :fit_ratios, :K, :log_constant, :n_bins_y, :tau]
    elseif c[:method_id] == "oqt"
        [:epsilon, :val_split]
    elseif c[:method_id] == "arc"
        [:with_binary_tree_regressor]
    elseif c[:method_id] in ["acc", "pacc"]
        [:val_split]
    elseif c[:method_id] in ["osld"]
        [:smoothing]
    elseif c[:method_id] in ["oacc", "opacc"]
        [:val_split, :criterion, :regularization, :tau]
    else
        Symbol[] # other methods (cc, pcc, sld) have no additional arguments
    end
    arg = Dict{Symbol,Any}() # extract the parameters from the configuration c
    for k in keys(c)
        if k in method_keys
            push!(arg, k => c[k])
        end
    end
    return arg
end

function _configure_classifier(config::Dict{Symbol,Any})
    classname = config[:classifier] # read the configuration
    parameters = haskey(config, :parameters) ? config[:parameters] : Dict{Symbol,Any}()
    preprocessing = get(config, :preprocessing, "")
    calibration = Symbol(get(config, :calibration, "none"))
    bagging = haskey(config, :bagging) ? config[:bagging] : Dict{Symbol,Any}()

    # instantiate classifier object
    Classifier = eval(Meta.parse(classname)) # constructor method
    classifier = Classifier(; parameters...)

    # add optional calibration
    if calibration == :isotonic
        classifier = CalibratedClassifierCV(
            classifier,
            method=string(calibration),
            cv=KFold(n_splits=3) # do not stratify CV
        )
    end

    # add optional bagging
    if length(bagging) > 0
        classifier = BaggingClassifier(classifier; bagging...)
    end

    # add optional pre-processing
    if preprocessing != ""
        transformer = eval(Meta.parse(preprocessing))() # call the constructor method
        classifier  = Pipeline([ ("preprocessing", transformer), ("classifier", classifier) ])
    end
    return classifier
end

# in-place substitutes of ScikitLearn.@sk_import
RandomForestClassifier(args...; kwargs...) = Util.SkObject("sklearn.ensemble.RandomForestClassifier", args...; kwargs...)
DecisionTreeClassifier(args...; kwargs...) = Util.SkObject("sklearn.tree.DecisionTreeClassifier", args...; kwargs...)
LogisticRegression(args...; kwargs...) = Util.SkObject("sklearn.linear_model.LogisticRegression", args...; kwargs...)
Pipeline(args...; kwargs...) = Util.SkObject("sklearn.pipeline.Pipeline", args...; kwargs...)
StandardScaler(args...; kwargs...) = Util.SkObject("sklearn.preprocessing.StandardScaler", args...; kwargs...)
BaggingClassifier(args...; kwargs...) = Util.SkObject("sklearn.ensemble.BaggingClassifier", args...; kwargs...)
CalibratedClassifierCV(args...; kwargs...) = Util.SkObject("sklearn.calibration.CalibratedClassifierCV", args...; kwargs...)
KFold(args...; kwargs...) = Util.SkObject("sklearn.model_selection.KFold", args...; kwargs...)
label_binarize(args...; kwargs...) = Util.SkObject("sklearn.preprocessing.label_binarize", args...; kwargs...)
IsotonicRegression(args...; kwargs...) = Util.SkObject("sklearn.isotonic.IsotonicRegression", args...; kwargs...)
CountVectorizer(args...; kwargs...) = Util.SkObject("sklearn.feature_extraction.text.CountVectorizer", args...; kwargs...)

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

_configure_stepsize(c::Dict{Symbol, Any}) =
    if c[:method_id] == "constant"
        alpha = get(c, :alpha, 1.0)
        ConstantStepsize(alpha)
    elseif c[:method_id] == "decay_mul"
        eta = c[:eta]
        a   = get(c, :a, 1.0)
        MulDecayStepsize(eta, a)   
    elseif c[:method_id] == "decay_exp"
        eta = c[:eta]
        a   = get(c, :a, 1.0)
        ExpDecayStepsize(eta, a)
    elseif c[:method_id] == "run"
        binning = _configure_binning(c[:binning])
        decay   = get(c, :decay, false)
        tau     = get(c, :tau, 0.0)
        warn    = get(c, :warn, false)
        RunStepsize(binning; decay = decay, tau = tau, warn = warn)
    else
        throw(ArgumentError("Unknown stepsize method_id=$(method_id)"))
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
    dirichlet([metaconfig = "conf/meta/dirichlet.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
function dirichlet(metaconfig::String="conf/meta/dirichlet.yml")
    meta = parsefile(metaconfig; dicttype=Dict{Symbol,Any})

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
            elseif clf[:classifier] in ["RandomForestClassifier", "DecisionTreeClassifier"]
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion]]
            end
            expand(clf, on...)
        end, pop!(job, :classifier))...))
        for clf in classifiers # interpolate classifier names
            if haskey(clf, :bagging)
                clf[:bagging][:random_state] = rand(UInt32)
            else
                clf[:parameters][:random_state] = rand(UInt32)
            end
            if clf[:classifier] == "LogisticRegression"
                clf[:name] = replace(clf[:name], "\$(C)" => clf[:parameters][:C])
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            elseif clf[:classifier] in ["RandomForestClassifier", "DecisionTreeClassifier"]
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
            if exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "sld"]
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
            elseif exp[:method_id] == "osld"
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:smoothing, :order],
                    [:smoothing, :impact],
                )
            elseif exp[:method_id] ∈ ["oacc", "opacc"]
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
            elseif exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "sld", "oqt", "arc", "acc", "pacc", "osld", "oacc", "opacc"]
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
                if exp[:method_id] in ["oqt", "arc", "acc", "pacc", "oacc", "opacc"]
                    name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:val_split]))}")
                    if exp[:method_id] ∈ ["oacc", "opacc"]
                        name = replace(name, "\$(val_split)" => exp[:val_split])
                        name = replace(name, "\$(criterion)" => Dict("mse"=>"L_2", "mae"=>"L_1")[exp[:criterion]])
                        name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:regularization]])
                        name = replace(name, "\$(tau)" => exp[:tau])
                    end
                elseif exp[:method_id] == "osld"
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
    amazon([metaconfig = "conf/meta/amazon.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
function amazon(metaconfig::String="conf/meta/amazon.yml")
    meta = parsefile(metaconfig; dicttype=Dict{Symbol,Any})

    # expand configuration
    for job in expand(meta, :data)
        interpolate!(job, :configfile, id=job[:data][:id])
        interpolate!(job, :outfile, id=job[:data][:id])
        interpolate!(job, :valfile, id=job[:data][:id])

        # expand classifiers, fix random seeds, and interpolate classifier names
        Random.seed!(job[:seed])
        classifiers = filter(clf -> begin # filtering: ignore LR on FACT
            job[:data][:id] != "tfidf" || clf[:classifier] ∉ ["RandomForestClassifier", "DecisionTreeClassifier"]
        end, vcat(map(clf -> begin # expansion
            on = if clf[:classifier] == "LogisticRegression"
                [[:parameters, :class_weight], [:parameters, :C]]
            elseif clf[:classifier] in ["RandomForestClassifier", "DecisionTreeClassifier"]
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion]]
            end
            expand(clf, on...)
        end, pop!(job, :classifier))...))
        for clf in classifiers # interpolate classifier names
            if haskey(clf, :bagging)
                clf[:bagging][:random_state] = rand(UInt32)
            else
                clf[:parameters][:random_state] = rand(UInt32)
            end
            if clf[:classifier] == "LogisticRegression"
                clf[:name] = replace(clf[:name], "\$(C)" => clf[:parameters][:C])
                if clf[:parameters][:class_weight] == "balanced"
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "u")
                else
                    clf[:name] = replace(clf[:name], "\$(class_weight)" => "n")
                end
            elseif clf[:classifier] in ["RandomForestClassifier", "DecisionTreeClassifier"]
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
            if exp[:method_id] in ["dsea", "dsea+", "cc", "pcc", "sld"]
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
            elseif exp[:method_id] == "osld"
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:smoothing, :order],
                    [:smoothing, :impact]
                )
            elseif exp[:method_id] ∈ ["oacc", "opacc"]
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
            elseif exp[:method_id] in ["oqt", "arc", "acc", "pacc", "oacc", "opacc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:val_split]))}")
                if exp[:method_id] ∈ ["oacc", "opacc"]
                    name = replace(name, "\$(val_split)" => exp[:val_split])
                    name = replace(name, "\$(criterion)" => Dict("mse"=>"L_2", "mae"=>"L_1")[exp[:criterion]])
                    name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:regularization]])
                    name = replace(name, "\$(tau)" => exp[:tau])
                end
            elseif exp[:method_id] == "osld"
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
