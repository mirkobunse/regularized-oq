module Job

using
    CSV,
    DataFrames,
    Discretizers,
    Distances,
    Distributed,
    Distributions,
    MLDataUtils,
    Optim,
    Printf,
    PyCall,
    Random,
    ScikitLearn,
    ScikitLearnBase
import LinearAlgebra, Statistics, StatsBase # import with qualified access, e.g. LinearAlgebra.eigen

using ComfyCommons: ComfyLogging, ComfyGit
using CherenkovDeconvolution
using CherenkovDeconvolution.OptimizedStepsizes: OptimizedStepsize
using CherenkovDeconvolution.DeconvUtil: expansion_discretizer, fit_pdf, fit_R, inspect_expansion, inspect_reduction, normalizepdf, train_and_predict_proba
using MetaConfigurations

using ..Util, ..Data
using ..Util: CalibratedRandomForestClassifier
using ..Conf: configure_method


# names for true spectra
TRUESPEC_IB  = "TRUE_SPECTRUM_IN_BAG"
TRUESPEC_OOB = "TRUE_SPECTRUM_OOB"
TRUESPEC     = "TRUE_SPECTRUM" # only use when no bootstrap is employed
TRAINSPEC    = "TRAIN_SPECTRUM"


"""
    run(configfile)

Hand the `configfile` to the Job method of which the name is configured by the property
`job` in the `configfile`.
"""
function run(configfile::String)
    c = parsefile(configfile)
    funname = "Job." * c["job"]
    
    @info "Calling $funname(\"$configfile\")"
    fun = eval(Meta.parse(funname))
    fun(configfile) # function call
end


"""
    pushseeds!(experiments, B; set_seed=true)

Add a random experiment seed and `B` seeds for the bootstrap samples to all configurations
in `experiments`. The bootstrap samples are equal in all experiments for comparability.

The experiment seed is stored in the property `seed`, the bootstrap seeds are an array in
the `bootstrap_seeds` property. When the global random number generator is seeded with
`Random.seed!()`, the result of this `pushseeds!()` is deterministic.
"""
function pushseeds!(experiments::AbstractArray{Dict{Symbol, Any}, 1}, B::Int; set_seed::Bool=true)
    bootstrapseeds = [ rand(UInt32) for _ in 1:B ]
    for exp in experiments
        exp[:bootstrap_seeds] = bootstrapseeds # equal in all experiments
        if set_seed
            exp[:seed] = rand(UInt32) # individual seed
        end
    end
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


# job implementations
include("job/smearing.jl")   # different amounts of smearing in data
include("job/clustering.jl") # the difficulty of classical deconvolution
include("job/svd.jl")        # re-formulations of deconvolution related to SVD
include("job/fit_ratios.jl") # fit ratios instead of pdfs
include("job/gridsearch.jl") # suitable parameters of a random forest on FACT data
include("job/classifier.jl") # different classifiers embedded in DSEA
include("job/weightfix.jl")  # corrected re-weighting of training examples
include("job/stepsize.jl")   # step size extension
include("job/smoothing.jl")  # smoothing extension
include("job/expand_reduce.jl") # expansion/reduction in DSEA
include("job/comparison.jl")    # comparative evaluation of RUN, IBU, and DSEA
include("job/time_series_contributions.jl") # contributions for time series analyses
include("job/smart_control.jl") # smart control of simulations with DSEA
include("job/feature_selection.jl") # impact of the number of selected features
include("job/uncertainty.jl") # estimate the uncertainty of deconvolution results

include("job/amazon.jl")
include("job/roberta.jl")
include("job/dirichlet.jl")


end

