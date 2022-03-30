module Res

using
    Colors,
    CriticalDifferenceDiagrams,
    CSV,
    DataFrames,
    Discretizers,
    Distances,
    HypothesisTests,
    MultivariateStats,
    PGFPlots,
    Printf,
    Query,
    Random,
    Statistics

using CherenkovDeconvolution
using CherenkovDeconvolution.DeconvUtil: fit_pdf, fit_R, normalizepdf

using ..Util, ..Data, ..Job
using ..Util: mdpa, nmd, rnod, rsnod

"""
    initialize(job, files...[; warn_changed=true])

Initialize the PGFPlots backend for the function Job.`job` called for the `files`.
"""
function initialize(job::String, files::String...; warn_changed=true)
    @warn "need to remove initialize"
end

include("res/amazon.jl")
include("res/main.jl")

end
