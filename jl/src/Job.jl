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

using CherenkovDeconvolution
using CherenkovDeconvolution.OptimizedStepsizes: OptimizedStepsize
using CherenkovDeconvolution.DeconvUtil: expansion_discretizer, fit_pdf, fit_R, inspect_expansion, inspect_reduction, normalizepdf, train_and_predict_proba
using MetaConfigurations
using ..Util, ..Data
using ..Conf: configure_method

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

include("job/amazon.jl")
include("job/dirichlet.jl")

end

