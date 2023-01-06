module Configuration

using CherenkovDeconvolution, MetaConfigurations, QUnfold, Random
using ..Util, ..Data, ..MoreMethods

DISCRETE_CONSTRUCTORS = Dict(
    "run" => CherenkovDeconvolution.RUN,
    "prun" => CherenkovDeconvolution.PRUN,
    "ibu" => CherenkovDeconvolution.IBU,
    "svd" => CherenkovDeconvolution.SVD
)
QUNFOLD_CONSTRUCTORS = Dict(
    "cc" => QUnfold.CC,
    "pcc" => QUnfold.PCC,
    "acc" => QUnfold.ACC,
    "pacc" => QUnfold.PACC,
    "sld" => QUnfold.SLD,
    "run" => QUnfold.RUN,
    "ibu" => QUnfold.IBU,
    "oacc" => QUnfold.ACC, # o-ACC uses the ACC constructor
    "opacc" => QUnfold.PACC, # o-PACC uses the PACC constructor
    "osld" => QUnfold.SLD, # o-SLD uses the SLD constructor
) # multi-class quantifiers from QuaPy
QUAPY_CONSTRUCTORS = Dict(
    "cc" => MoreMethods.ClassifyAndCount,
    "pcc" => MoreMethods.ProbabilisticClassifyAndCount,
    "acc" => MoreMethods.AdjustedClassifyAndCount,
    "pacc" => MoreMethods.ProbabilisticAdjustedClassifyAndCount,
    "sld" => MoreMethods.ExpectationMaximizationQuantifier,
) # multi-class quantifiers from QuaPy

"""
    configure_method(c::Dict{Symbol, Any})

Set up an OQ method from CherenkovDeconvolution.jl or from src/MoreMethods.jl.
"""
function configure_method(c::Dict{Symbol, Any})
    c = copy(c) # keep the input unchanged; CAUTION: this copy is only shallow

    # initialize classifiers, binnings, and methods
    if c[:method_id] == "oqt"
        return MoreMethods.OQT(_configure_classifier(c[:classifier]); c[:parameters]...)
    elseif c[:method_id] == "arc"
        return MoreMethods.ARC(_configure_classifier(c[:classifier]); c[:parameters]...)
    elseif c[:method_id] in keys(QUNFOLD_CONSTRUCTORS)
        constructor = QUNFOLD_CONSTRUCTORS[c[:method_id]]
        args = Any[] # set up positional arguments
        if haskey(c, :transformer)
            if c[:transformer] == "classifier"
                push!(args, QUnfold.ClassTransformer(_configure_classifier(c[:classifier])))
            elseif c[:transformer] == "tree"
                push!(args, QUnfold.TreeTransformer(DecisionTreeClassifier(; c[:transformer_parameters]...)))
            end
        else
            if haskey(c, :classifier) && c[:method_id] ∉ ["hdx", "ohdx"]
                push!(args, _configure_classifier(c[:classifier]))
            end
            if haskey(c, :parameters) && haskey(c[:parameters], :n_bins)
                c[:parameters] = copy(c[:parameters]) # a shallow copy in the deep
                push!(args, pop!(c[:parameters], :n_bins))
            end
        end
        kwargs = get(c, :parameters, Dict{Symbol,Any}())
        try
            return constructor(args...; kwargs...)
        catch
            @error "Cannot configure method" c[:method_id] args kwargs
            rethrow()
        end
    else
        throw(ArgumentError("Unknown method_id=$(c[:method_id])"))
    end
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

        # expand experiments
        J_seed = Dict(map( # make sure that each TreeTransformer is the same
            J -> J => rand(UInt32),
            unique(vcat(MetaConfigurations.find(job[:method], :max_leaf_nodes)...))
        )...)
        job[:method] = vcat(map(exp -> begin # expansion
            if exp[:method_id] in ["cc", "pcc", "acc", "pacc", "sld"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc"]
                exp[:classifier] = classifiers
                expand(exp, [:parameters, :val_split], :classifier)
            elseif exp[:method_id] == "ibu"
                expand(exp,
                    [:parameters, :o],
                    [:parameters, :λ],
                    [:transformer_parameters, :max_leaf_nodes]
                )
            elseif exp[:method_id] in ["run", "svd"]
                expand(exp, [:parameters, :τ], [:transformer_parameters, :max_leaf_nodes])
            elseif exp[:method_id] == "osld"
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:parameters, :o],
                    [:parameters, :λ],
                )
            elseif exp[:method_id] ∈ ["oacc", "opacc"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ]) # :regularization
            else
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...)

        # interpolate the method names and seed the TreeTransformers
        for exp in job[:method]
            name = exp[:name]
            if exp[:method_id] in ["ibu", "run", "svd"]
                seed = J_seed[exp[:transformer_parameters][:max_leaf_nodes]]
                exp[:transformer_parameters][:random_state] = seed
                name = replace(name, "\$(max_leaf_nodes)" => exp[:transformer_parameters][:max_leaf_nodes])
            end
            if haskey(exp, :classifier)
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
            end
            if exp[:method_id] in ["ibu", "osld"]
                name = replace(name, "\$(o)" => exp[:parameters][:o])
                name = replace(name, "\$(λ)" => exp[:parameters][:λ])
            elseif exp[:method_id] in ["run", "svd", "oacc", "opacc"]
                name = replace(name, "\$(τ)" => exp[:parameters][:τ])
                # if exp[:method_id] ∈ ["oacc", "opacc"]
                #     name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:parameters][:regularization]])
                # end
            elseif exp[:method_id] in ["oqt", "arc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:parameters][:val_split]))}")
            end
            exp[:name] = name # replace with interpolation
        end

        # collect experiments per method_id
        method_exp = Dict{String,Vector{Dict{Symbol,Any}}}()
        for exp in job[:method]
            id = exp[:method_id]
            method_exp[id] = push!(get(method_exp, id, Dict{Symbol,Any}[]), exp)
        end
        for (method, exp) in pairs(method_exp)
            @info "$(method) will be optimized over $(length(exp)) configurations"
        end

        # write the generated job configuration to a file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)

        # derive a testing configuration
        for x in [:configfile, :outfile, :valfile]
            job[x] = joinpath(dirname(job[x]), "test_" * basename(job[x]))
        end
        job[:M_val] = 3
        job[:M_tst] = 3
        job[:N_tst] = 100
        job[:N_val] = 100
        job[:N_trn] = 1000
        job[:protocol][:n_splits] = 2
        job[:method] = vcat(rand.(values(method_exp))...)
        @info "Writing a test configuration to $(job[:configfile])"
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
        classifiers = vcat(map(clf -> begin # expansion
            on = if clf[:classifier] == "LogisticRegression"
                [[:parameters, :class_weight], [:parameters, :C]]
            elseif clf[:classifier] in ["RandomForestClassifier", "DecisionTreeClassifier"]
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion]]
            end
            expand(clf, on...)
        end, pop!(job, :classifier))...)
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
        job[:method] = vcat(map(exp -> begin # expansion
            exp = deepcopy(exp)
            if exp[:method_id] in ["cc", "pcc", "acc", "pacc", "sld"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc"]
                exp[:classifier] = classifiers
                expand(exp, [:parameters, :val_split], :classifier)
            elseif exp[:method_id] in ["ibu", "osld"]
                exp[:classifier] = classifiers
                expand(exp,
                    :classifier,
                    [:parameters, :o],
                    [:parameters, :λ]
                )
            elseif exp[:method_id] in ["run", "svd"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ])
            elseif exp[:method_id] ∈ ["oacc", "opacc"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ]) # :regularization
            else # other methods are not supported
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...)

        # interpolate the method names
        for exp in job[:method]
            name = exp[:name]
            if haskey(exp, :classifier)
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
            end
            if exp[:method_id] in ["ibu", "osld"]
                name = replace(name, "\$(o)" => exp[:parameters][:o])
                name = replace(name, "\$(λ)" => exp[:parameters][:λ])
            elseif exp[:method_id] in ["run", "svd", "oacc", "opacc"]
                name = replace(name, "\$(τ)" => exp[:parameters][:τ])
                # if exp[:method_id] ∈ ["oacc", "opacc"]
                #     name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:parameters][:regularization]])
                # end
            elseif exp[:method_id] in ["oqt", "arc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:parameters][:val_split]))}")
            end
            exp[:name] = name # replace with interpolation
        end

        # collect experiments per method_id
        method_exp = Dict{String,Vector{Dict{Symbol,Any}}}()
        for exp in job[:method]
            id = exp[:method_id]
            method_exp[id] = push!(get(method_exp, id, Dict{Symbol,Any}[]), exp)
        end
        for (method, exp) in pairs(method_exp)
            @info "$(method) will be optimized over $(length(exp)) configurations"
        end

        # write job to file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)

        # derive a testing configuration
        for x in [:configfile, :outfile, :valfile]
            job[x] = joinpath(dirname(job[x]), "test_" * basename(job[x]))
        end
        job[:M_val] = 3
        job[:M_tst] = 3
        job[:N_trn] = 1000
        job[:protocol][:n_splits] = 2
        job[:method] = vcat(rand.(values(method_exp))...)
        @info "Writing a test configuration to $(job[:configfile])"
        save(job[:configfile], job)
    end
end

end # module
