module Configuration

using
    CherenkovDeconvolution,
    Distances,
    MetaConfigurations,
    OrderedCollections,
    PyCall,
    QUnfold,
    Random
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
    "hdx" => QUnfold.HDx,
    "hdy" => QUnfold.HDy,
    "run" => QUnfold.RUN,
    "ibu" => QUnfold.IBU,
    "svd" => QUnfold.SVD,
    "oacc" => QUnfold.ACC, # o-ACC uses the ACC constructor
    "opacc" => QUnfold.PACC, # o-PACC uses the PACC constructor
    "osld" => QUnfold.SLD, # o-SLD uses the SLD constructor
    "ohdx" => QUnfold.HDx, # o-HDx uses the HDx constructor
    "ohdy" => QUnfold.HDy, # o-HDy uses the HDy constructor
    "edy" => QUnfold.EDy,
    "pdf" => QUnfold.PDF,
) # multi-class quantifiers from QuaPy
QUAPY_CONSTRUCTORS = Dict(
    "quapy-cc" => MoreMethods.ClassifyAndCount,
    "quapy-pcc" => MoreMethods.ProbabilisticClassifyAndCount,
    "quapy-acc" => MoreMethods.AdjustedClassifyAndCount,
    "quapy-pacc" => MoreMethods.ProbabilisticAdjustedClassifyAndCount,
    "quapy-sld" => MoreMethods.ExpectationMaximizationQuantifier,
) # multi-class quantifiers from QuaPy
CASTANO_CONSTRUCTORS = Dict(
    "castano-cc" => MoreMethods.CastanoCC,
    "castano-pcc" => MoreMethods.CastanoPCC,
    "castano-acc" => MoreMethods.CastanoAC,
    "castano-pacc" => MoreMethods.CastanoPAC,
    "castano-edy" => MoreMethods.CastanoEDy,
    "castano-pdf" => MoreMethods.CastanoPDF,
) # ordinal quantifiers from https://github.com/mirkobunse/ordinal_quantification

function __init__() # copy the CVClassifier from qunfold.sklearn
    py"""
    import numpy as np
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.exceptions import NotFittedError
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.multiclass import unique_labels
    class CVClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, estimator, n_estimators, random_state=None):
            self.estimator = estimator
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.oob_score = True # the whole point of this class is to have an oob_score
        def fit(self, X, y):
            self.estimators_ = []
            self.i_classes_ = [] # the indices of each estimator's subset of classes
            self.classes_ = unique_labels(y)
            self.oob_decision_function_ = np.zeros((len(y), len(self.classes_)))
            class_mapping = dict(zip(self.classes_, np.arange(len(self.classes_))))
            skf = StratifiedKFold(
                n_splits = self.n_estimators,
                random_state = self.random_state,
                shuffle = True
            )
            for i_trn, i_tst in skf.split(X, y):
                estimator = clone(self.estimator).fit(X[i_trn], y[i_trn])
                i_classes = np.array([ class_mapping[_class] for _class in estimator.classes_ ])
                y_pred = estimator.predict_proba(X[i_tst])
                self.oob_decision_function_[i_tst[:, np.newaxis], i_classes[np.newaxis, :]] = y_pred
                self.estimators_.append(estimator)
                self.i_classes_.append(i_classes)
            return self
        def predict_proba(self, X):
            if not hasattr(self, "classes_"):
                raise NotFittedError()
            y_pred = np.zeros((len(self.estimators_), len(X), len(self.classes_)))
            for i, (estimator, i_classes) in enumerate(zip(self.estimators_, self.i_classes_)):
                y_pred[i, :, i_classes] = estimator.predict_proba(X).T
            return np.mean(y_pred, axis=0) # shape (n_samples, n_classes)
        def predict(self, X):
            y_pred = self.predict_proba(X).argmax(axis=1) # class indices
            return self.classes_[y_pred]
    """
end
CVClassifier(estimator; n_estimators::Int=10, random_state::Int=Int(rand(UInt32))) =
    py"CVClassifier"(estimator, n_estimators, random_state)

"""
    configure_method(c::Dict{Symbol, Any}[, classifier=nothing])

Set up an OQ method from CherenkovDeconvolution.jl or from src/MoreMethods.jl.
"""
function configure_method(c::Dict{Symbol, Any}, classifier::Any=nothing)
    c = copy(c) # keep the input unchanged; CAUTION: this copy is only shallow
    kwargs = copy(get(c, :parameters, Dict{Symbol,Any}())) # a shallow copy in the deep
    if classifier === nothing && haskey(c, :classifier)
        classifier = configure_classifier(c[:classifier])
    end

    # initialize classifiers, binnings, and methods
    if c[:method_id] == "oqt"
        return MoreMethods.OQT(classifier; kwargs...)
    elseif c[:method_id] == "arc"
        return MoreMethods.ARC(classifier; kwargs...)
    elseif c[:method_id] in keys(QUNFOLD_CONSTRUCTORS)
        constructor = QUNFOLD_CONSTRUCTORS[c[:method_id]]
        args = Any[] # set up positional arguments
        if haskey(c, :transformer)
            if c[:transformer] == "classifier"
                push!(args, QUnfold.ClassTransformer(classifier; fit_classifier=pop!(kwargs, :fit_classifier, true)))
            elseif c[:transformer] == "tree"
                push!(args, QUnfold.TreeTransformer(
                    DecisionTreeClassifier(; c[:transformer_parameters][:tree_parameters]...);
                    fit_tree = c[:transformer_parameters][:fit_tree]
                ))
            end
        else
            if classifier !== nothing && c[:method_id] ∉ ["hdx", "ohdx"]
                push!(args, classifier)
            end
            if haskey(kwargs, :n_bins)
                push!(args, pop!(kwargs, :n_bins))
            end
        end
        if haskey(kwargs, :strategy)
            kwargs[:strategy] = Symbol(kwargs[:strategy])
        end
        if haskey(kwargs, :distance)
            kwargs[:distance] = Dict(
                "Euclidean" => Euclidean(),
                "EarthMovers" => QUnfold.EarthMovers(),
                "EarthMoversSurrogate" => QUnfold.EarthMoversSurrogate(),
            )[kwargs[:distance]]
        end
        try
            return constructor(args...; kwargs...)
        catch
            @error "Cannot configure method" c[:method_id] args kwargs
            rethrow()
        end
    elseif c[:method_id] in keys(QUAPY_CONSTRUCTORS)
        return QUAPY_CONSTRUCTORS[c[:method_id]](classifier; kwargs...)
    elseif c[:method_id] in keys(CASTANO_CONSTRUCTORS)
        args = Any[ classifier ] # set up positional arguments
        if haskey(kwargs, :n_bins)
            push!(args, pop!(kwargs, :n_bins))
        end
        if haskey(kwargs, :distances)
            if kwargs[:distances] == "emd_distances"
                kwargs[:distances] = MoreMethods.__castano_emd_distances
            elseif kwargs[:distances] == "euclidean_distances"
                kwargs[:distances] = MoreMethods.__sklearn_euclidean_distances
            end
        end
        if get(kwargs, :decomposer, "monotone") == "monotone"
            kwargs[:decomposer] = MoreMethods.__castano_factory.Decomposer.monotone
        elseif get(kwargs, :decomposer, "monotone") == "none"
            kwargs[:decomposer] = MoreMethods.__castano_factory.Decomposer.none
        end
        if get(kwargs, :option, "cv_decomp") == "cv_decomp"
            kwargs[:option] = MoreMethods.__castano_factory.Option.cv_decomp
        elseif get(kwargs, :option, "cv_decomp") == "bagging_decomp"
            kwargs[:option] = MoreMethods.__castano_factory.Option.bagging_decomp
        end
        return CASTANO_CONSTRUCTORS[c[:method_id]](args...; kwargs...)
    else
        throw(ArgumentError("Unknown method_id=$(c[:method_id])"))
    end
end

function configure_classifier(config::Dict{Symbol,Any})
    classname = config[:classifier] # read the configuration
    parameters = haskey(config, :parameters) ? config[:parameters] : Dict{Symbol,Any}()
    preprocessing = get(config, :preprocessing, "")
    calibration = Symbol(get(config, :calibration, "none"))
    bagging = haskey(config, :bagging) ? config[:bagging] : Dict{Symbol,Any}()
    cv = haskey(config, :cv) ? config[:cv] : Dict{Symbol,Any}()

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
    elseif length(cv) > 0
        classifier = CVClassifier(classifier; cv...)
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
    dirichlet(metaconfigs=["conf/meta/dirichlet_fact.yml", "conf/meta/dirichlet_others.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
dirichlet(metaconfigs::Vararg{String,N}=("conf/meta/dirichlet_fact.yml", "conf/meta/dirichlet_others.yml")...) where {N} = for m in metaconfigs _dirichlet(m) end

function _dirichlet(metaconfig::String)
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
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion], [:parameters, :min_samples_leaf]]
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
                clf[:name] = replace(clf[:name], "\$(min_samples_leaf)" => clf[:parameters][:min_samples_leaf])
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
        job[:method] = vcat(map(exp -> begin # expansion
            if exp[:method_id] in ["cc", "pcc", "acc", "pacc", "sld"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] == "castano-pdf"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :n_bins])
            elseif exp[:method_id] in keys(CASTANO_CONSTRUCTORS)
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc"]
                exp[:classifier] = classifiers
                expand(exp, [:parameters, :val_split], :classifier)
            elseif exp[:method_id] == "ibu"
                if exp[:transformer] == "tree"
                    filter(
                        x -> x[:parameters][:λ] > 0 || x[:parameters][:o] == 0,
                        expand(exp,
                            [:parameters, :o],
                            [:parameters, :λ],
                            [:transformer_parameters, :tree_parameters, :max_leaf_nodes],
                            [:transformer_parameters, :tree_parameters, :class_weight],
                            [:transformer_parameters, :fit_tree]
                        )
                    )
                elseif exp[:transformer] == "classifier"
                    exp[:classifier] = classifiers
                    filter(
                        x -> x[:parameters][:λ] > 0 || x[:parameters][:o] == 0,
                        expand(exp, [:parameters, :o], [:parameters, :λ], :classifier)
                    )
                end
            elseif exp[:method_id] in ["run", "svd"]
                if exp[:transformer] == "tree"
                    expand(exp,
                        [:parameters, :τ],
                        [:transformer_parameters, :tree_parameters, :max_leaf_nodes],
                        [:transformer_parameters, :tree_parameters, :class_weight],
                        [:transformer_parameters, :fit_tree]
                    )
                elseif exp[:transformer] == "classifier"
                    exp[:classifier] = classifiers
                    expand(exp, [:parameters, :τ], :classifier)
                end
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
            elseif exp[:method_id] == "hdx"
                expand(exp, [:parameters, :n_bins])
            elseif exp[:method_id] == "hdy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :n_bins])
            elseif exp[:method_id] == "ohdx"
                expand(exp, [:parameters, :τ], [:parameters, :n_bins])
            elseif exp[:method_id] == "ohdy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ], [:parameters, :n_bins])
            elseif exp[:method_id] == "edy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ])
            elseif exp[:method_id] == "pdf"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ], [:parameters, :n_bins])
            else
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...)

        # interpolate the method names and provide consistent seeds for TreeTransformer & HDx
        tree_seed = Dict(map( # make sure that each TreeTransformer is the same
            J -> J => rand(UInt32),
            unique(vcat(MetaConfigurations.find(job[:method], :max_leaf_nodes)...))
        )...)
        hdx_seed = Dict(map( # make sure that each HDx n_bins setting is the same
            J -> J => rand(UInt32),
            unique(vcat(MetaConfigurations.find(job[:method], :n_bins)...))
        )...)
        for exp in job[:method]
            name = exp[:name]
            if get(exp, :transformer, "") == "tree"
                seed = tree_seed[exp[:transformer_parameters][:tree_parameters][:max_leaf_nodes]]
                exp[:transformer_parameters][:tree_parameters][:random_state] = seed
                name = replace(name, "\$(max_leaf_nodes)" => exp[:transformer_parameters][:tree_parameters][:max_leaf_nodes])
                if exp[:transformer_parameters][:tree_parameters][:class_weight] == "balanced"
                    name = replace(name, "\$(class_weight)" => "u")
                else
                    name = replace(name, "\$(class_weight)" => "n")
                end
                name = replace(name, "\$(fit_tree)" => exp[:transformer_parameters][:fit_tree])
            end
            if haskey(exp, :classifier)
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
            end
            if haskey(exp, :parameters) && haskey(exp[:parameters], :τ)
                name = replace(name, "\$(τ)" => exp[:parameters][:τ])
            end
            if exp[:method_id] in ["ibu", "osld"]
                name = replace(name, "\$(o)" => exp[:parameters][:o])
                name = replace(name, "\$(λ)" => exp[:parameters][:λ])
            # elseif exp[:method_id] ∈ ["oacc", "opacc"]
            #     name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:parameters][:regularization]])
            elseif exp[:method_id] in ["oqt", "arc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:parameters][:val_split]))}")
            elseif exp[:method_id] in ["hdx", "hdy", "ohdx", "ohdy", "pdf", "castano-pdf"]
                name = replace(name, "\$(n_bins)" => exp[:parameters][:n_bins])
                if exp[:method_id] in ["hdx", "ohdx"]
                    exp[:random_state] = hdx_seed[exp[:parameters][:n_bins]]
                end
            end
            exp[:name] = name # replace with interpolation
        end

        # collect experiments per validation_group
        group_exp = Dict{String,Vector{Dict{Symbol,Any}}}()
        for exp in job[:method]
            validation_group = get(exp, :validation_group, exp[:method_id])
            group_exp[validation_group] = push!(
                get(group_exp, validation_group, Dict{Symbol,Any}[]),
                exp
            )
        end
        for (validation_group, exp) in pairs(group_exp)
            @info "$(validation_group) will be optimized over $(length(exp)) configurations"
        end

        # write the generated job configuration to a file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)

        # derive a testing configuration
        for x in [:configfile, :outfile, :valfile]
            job[x] = joinpath(dirname(job[x]), "test_" * basename(job[x]))
        end
        job[:M_val] = 5
        job[:M_tst] = 5
        job[:N_tst] = 100
        job[:N_val] = 100
        job[:N_trn] = 2000
        job[:method] = vcat(rand.(values(group_exp))...)
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
                [[:parameters, :class_weight], [:parameters, :max_depth], [:parameters, :criterion], [:parameters, :min_samples_leaf]]
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
                clf[:name] = replace(clf[:name], "\$(min_samples_leaf)" => clf[:parameters][:min_samples_leaf])
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
        job[:method] = vcat(map(exp -> begin # expansion
            exp = deepcopy(exp)
            if exp[:method_id] in ["cc", "pcc", "acc", "pacc", "sld", "quapy-sld"]
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] == "castano-pdf"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :n_bins])
            elseif exp[:method_id] in keys(CASTANO_CONSTRUCTORS)
                exp[:classifier] = classifiers
                expand(exp, :classifier)
            elseif exp[:method_id] in ["oqt", "arc"]
                exp[:classifier] = classifiers
                expand(exp, [:parameters, :val_split], :classifier)
            elseif exp[:method_id] in ["ibu", "osld"]
                exp[:classifier] = classifiers
                filter(
                    x -> x[:parameters][:λ] > 0 || x[:parameters][:o] == 0,
                    expand(exp,
                        :classifier,
                        [:parameters, :o],
                        [:parameters, :λ]
                    )
                )
            elseif exp[:method_id] in ["run", "svd"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ])
            elseif exp[:method_id] ∈ ["oacc", "opacc"]
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ]) # :regularization
            elseif exp[:method_id] == "hdx"
                expand(exp, [:parameters, :n_bins])
            elseif exp[:method_id] == "hdy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :n_bins])
            elseif exp[:method_id] == "ohdx"
                expand(exp, [:parameters, :τ], [:parameters, :n_bins])
            elseif exp[:method_id] == "ohdy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ], [:parameters, :n_bins])
            elseif exp[:method_id] == "edy"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ])
            elseif exp[:method_id] == "pdf"
                exp[:classifier] = classifiers
                expand(exp, :classifier, [:parameters, :τ], [:parameters, :n_bins])
            else # other methods are not supported
                throw(ArgumentError("Illegal method $(exp[:method_id])"))
            end
        end, job[:method])...)

        # interpolate the method names and provide consistent seeds for HDx
        hdx_seed = Dict(map( # make sure that each HDx n_bins setting is the same
            J -> J => rand(UInt32),
            unique(vcat(MetaConfigurations.find(job[:method], :n_bins)...))
        )...)
        for exp in job[:method]
            name = exp[:name]
            if haskey(exp, :classifier)
                name = replace(name, "\$(classifier)" => exp[:classifier][:name])
            end
            if haskey(exp, :parameters) && haskey(exp[:parameters], :τ)
                name = replace(name, "\$(τ)" => exp[:parameters][:τ])
            end
            if exp[:method_id] in ["ibu", "osld"]
                name = replace(name, "\$(o)" => exp[:parameters][:o])
                name = replace(name, "\$(λ)" => exp[:parameters][:λ])
            # elseif exp[:method_id] ∈ ["oacc", "opacc"]
            #     name = replace(name, "\$(regularization)" => Dict("curvature"=>"C_2", "difference"=>"C_1", "norm"=>"I")[exp[:parameters][:regularization]])
            elseif exp[:method_id] in ["oqt", "arc"]
                name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:parameters][:val_split]))}")
            elseif exp[:method_id] in ["hdx", "hdy", "ohdx", "ohdy", "pdf", "castano-pdf"]
                name = replace(name, "\$(n_bins)" => exp[:parameters][:n_bins])
                if exp[:method_id] in ["hdx", "ohdx"]
                    exp[:random_state] = hdx_seed[exp[:parameters][:n_bins]]
                end
            end
            exp[:name] = name # replace with interpolation
        end

        if job[:data][:type] in [ "raw_text", "tfidf" ]
            job[:method] = filter( # omit HDx, ARC, and OQT (sparse matrices cannot be split)
                exp -> exp[:method_id] ∉ [ "hdx", "ohdx", "arc", "oqt" ],
                job[:method]
            )
        end

        # collect experiments per validation_group
        group_exp = Dict{String,Vector{Dict{Symbol,Any}}}()
        for exp in job[:method]
            validation_group = get(exp, :validation_group, exp[:method_id])
            group_exp[validation_group] = push!(
                get(group_exp, validation_group, Dict{Symbol,Any}[]),
                exp
            )
        end
        for (validation_group, exp) in pairs(group_exp)
            @info "$(validation_group) will be optimized over $(length(exp)) configurations"
        end

        # write job to file
        @info "Writing configuration of $(length(job[:method])) experiments to $(job[:configfile])"
        save(job[:configfile], job)

        # derive a testing configuration
        for x in [:configfile, :outfile, :valfile]
            job[x] = joinpath(dirname(job[x]), "test_" * basename(job[x]))
        end
        job[:M_val] = 5
        job[:M_tst] = 5
        job[:N_trn] = 2000
        job[:method] = vcat(rand.(values(group_exp))...)
        @info "Writing a test configuration to $(job[:configfile])"
        save(job[:configfile], job)
    end
end

"""
    castano([metaconfig = "conf/meta/castano.yml"])

Generate a set of job configurations from the given meta-configuration file.
"""
function castano(metaconfig::String="conf/meta/castano.yml")
    meta = parsefile(metaconfig; dicttype=OrderedDict{Symbol,Any})

    # expand methods
    meta[:method] = vcat(map(exp -> begin
        exp = deepcopy(exp)
        if exp[:method_id] in ["hdx", "hdy"]
            expand(exp, [:parameters, :n_bins])
        elseif exp[:method_id] in ["oqt", "arc"]
            expand(exp, [:parameters, :val_split])
        elseif exp[:method_id] in ["ibu", "osld"]
            expand(exp, [:parameters, :o], [:parameters, :λ])
        elseif exp[:method_id] in ["run", "svd", "oacc", "opacc", "edy"]
            expand(exp, [:parameters, :τ]) # :regularization
        elseif exp[:method_id] in ["ohdx", "ohdy", "pdf"]
            expand(exp, [:parameters, :τ], [:parameters, :n_bins])
        elseif exp[:method_id] in ["cc", "pcc", "acc", "pacc", "sld", "quapy-sld"]
            exp
        else
            throw(ArgumentError("Illegal method $(exp[:method_id])"))
        end
    end, meta[:method])...)

    # interpolate the method names and provide consistent seeds for HDx
    hdx_seed = Dict(map( # make sure that each HDx n_bins setting is the same
        J -> J => rand(UInt32),
        unique(vcat(MetaConfigurations.find(meta[:method], :n_bins)...))
    )...)
    for exp in meta[:method]
        name = exp[:name]
        if haskey(exp, :parameters) && haskey(exp[:parameters], :τ)
            name = replace(name, "\$(τ)" => exp[:parameters][:τ])
        end
        if exp[:method_id] in ["ibu", "osld"]
            name = replace(name, "\$(o)" => exp[:parameters][:o])
            name = replace(name, "\$(λ)" => exp[:parameters][:λ])
        elseif exp[:method_id] in ["oqt", "arc"]
            name = replace(name, "\$(val_split)" => "\\frac{1}{$(round(Int, 1/exp[:parameters][:val_split]))}")
        elseif exp[:method_id] in ["hdx", "hdy", "ohdx", "ohdy", "pdf"]
            name = replace(name, "\$(n_bins)" => exp[:parameters][:n_bins])
            if exp[:method_id] in ["hdx", "ohdx"]
                exp[:random_state] = hdx_seed[exp[:parameters][:n_bins]]
            end
        end
        exp[:name] = name # replace with interpolation
    end

    # collect experiments per validation_group
    group_exp = Dict{String,Vector{Dict{Symbol,Any}}}()
    for exp in meta[:method]
        validation_group = get(exp, :validation_group, exp[:method_id])
        group_exp[validation_group] = push!(
            get(group_exp, validation_group, Dict{Symbol,Any}[]),
            exp
        )
    end
    for (validation_group, exp) in pairs(group_exp)
        @info "$(validation_group) will be optimized over $(length(exp)) configurations"
    end

    # write the generated job configuration to a file
    @info "Writing a configuration of $(length(meta[:method])) methods to $(meta[:configfile])"
    save(meta[:configfile], meta)
end

end # module
