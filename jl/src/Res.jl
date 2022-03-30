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

using ComfyCommons: ComfyLogging, ComfyGit
using CherenkovDeconvolution
using CherenkovDeconvolution.DeconvUtil: fit_pdf, fit_R, normalizepdf

using ..Util, ..Data, ..Job
using ..Util: kl, chi2p, chi2s, mdpa, nmd, rnod, rsnod

"""
    initialize(job, files...[; warn_changed=true])

Initialize the PGFPlots backend for the function Job.`job` called for the `files`.
"""
function initialize(job::String, files::String...; warn_changed=true)
    
    # warn about changes
    commit     = ComfyGit.commithash()
    remote     = ComfyGit.remoteurl()
    haschanges = ComfyGit.haschanges("src", files...)
    if haschanges && warn_changed
        @warn "Uncommited changes may affect plots"
    end
    
    # set preamble for PDF output
    preamble = join(readlines("src/res/tex/preamble.tex", keep=true)) # basic preamble
    preamble = foldl(replace, [ "GitCommit={}" =>            "GitCommit={$commit}",
                                "GitOrigin={}" =>            "GitOrigin={$remote}",
                                "GitUncommitedChanges={}" => "GitUncommitedChanges={$haschanges}" ], init=preamble)
    resetPGFPlotsPreamble()
    pushPGFPlotsPreamble(preamble)
    
    @warn "Need to include git information in the tex file"
    # # set file template
    # ComfyPgf.setpgftemplate("""
    # % 
    # % This file was generated with Res.$job(\"$(join(files, "\""))\")
    # % 
    # % git commit = $commit
    # % git origin = $remote
    # % uncommited changes = $haschanges
    # % 
    # $(ComfyPgf.TEMPLATE_PLOT_PLACEHOLDER)
    # """)
    
end


METRICS = Dict( :kl    => kl, # function from DeconvExp.Util
                :h     => Distances.hellinger,
                :chi2p => chi2p,
                :chi2s => chi2s,
                :emd   => mdpa,
                :nmd   => nmd,
                :rnod  => rnod,
                :rsnod => rsnod )

METRIC_NAMES = Dict( :kl    => "Kullback-Leibler distance",
                     :h     => "Hellinger distance",
                     :chi2p => "\$\\mathcal{X}^2_P\$",
                     :chi2s => "\$\\mathcal{X}^2_{Sym}\$",
                     :emd   => "EMD",
                     :nmd   => "NMD",
                     :rnod  => "RNOD",
                     :rsnod => "RSNOD" )

DEFAULT_METRIC = :emd # omit the others by default
keys_metrics(all_metrics::Bool) = all_metrics ? keys(METRICS) : [ DEFAULT_METRIC ]

"""
    all_metrics(regex=r".*")

Generate metrics files for all results in `res/spectra` matching the regex.
"""
function all_metrics(regex::Regex=r".*")
    spectradir = "res/spectra" # find files in spectra directory
    files = filter(readdir(spectradir)) do f
        !startswith(f, "TEST-") && endswith(f, ".csv") && occursin(regex, f)
    end
    @info "Computing metrics from all files matching $spectradir/$regex" files
    
    map(files) do f # compute metrics for each file
        metrics(joinpath(spectradir, f))
    end
    return nothing
end

"""
    metrics(spectrafile)

Evaluate the deconvolution results stored in the `spectrafile`, writing to a corresponding
metrics file.
"""
function metrics(spectrafile::String; eval_log_a::Bool = false, log_constant::Number = 183941)
    jobname = _job_name(spectrafile)

    # read file
    @info "Reading $spectrafile of the job $jobname"
    df = coalesce.(CSV.read(spectrafile, DataFrame), "") # replace missing Strings with ""

    # columns to join on
    jcols = get(JCOLS, jobname, [:b]) # join on :b, by default
    @info "Will join results on $jcols"

    # set up acceptance correction
    if eval_log_a
        _ , inv_ac = Data.acceptance_correction(Data.Fact())
        a = inv_ac(ones(12))
        log_a = f -> DeconvUtil.normalizepdf(log.(1 .+ a .* f .* log_constant))
    end

    # split into reference spectra and deconvolution results
    refs = Dict(
        "ib"  => unique(df[df[!, :name] .== Job.TRUESPEC_IB,  vcat(jcols, :f)]),
        "oob" => unique(df[df[!, :name] .== Job.TRUESPEC_OOB, vcat(jcols, :f)]),
        "" => unique(df[df[!, :name] .== Job.TRUESPEC, vcat(jcols, :f)])
    )
    for (id, ref) in refs
        if nrow(ref) > 0
            ref[!, :f] = [ normalizepdf(eval.(Meta.parse.(ref[!, :f]))...)... ]
            if eval_log_a
                ref[!, :f] = map(x -> log_a(x), ref[!, :f])
            end
            @info "Found $(size(ref, 1)) unique $id reference spectra - expecting 20 for each data set"
        else
            @info "Found no $(id != "" ? id : "non-bootstrap") reference spectra"
        end
    end
    df = df[.&(df[!, :name] .!= Job.TRUESPEC_IB,
               df[!, :name] .!= Job.TRUESPEC_OOB,
               df[!, :name] .!= Job.TRUESPEC), :]

    # special pre-processing of results
    if jobname == "comparison" # other experiments do not need this
        @info "Selecting relevant results.."
        df = _metrics_comparison(df)
    elseif jobname == "classifier" # other experiments do not need this
        @info "Selecting relevant results.."
        df = _metrics_classifier(df)
    end
    df = unique(df)

    # parse and normalize arrays
    @info "Parsing $(size(df, 1)) spectra.."
    df[!, :f] = [ normalizepdf(eval.(Meta.parse.(df[!, :f]))...)... ]
    if eval_log_a
      df[!, :f] = map(x -> log_a(x), df[!, :f])
    end

    # evaluate metrics with respect to both reference spectra
    for (id, ref) in refs
        if nrow(ref) > 0
            # join results with reference
            rename!(ref, :f => :f_ref)
            df = outerjoin(df, ref, on = jcols) # join on jcols

            # evaluate metrics
            for (mkey, mfun) in METRICS
                mcol = id != "" ? Symbol("$(id)_$(string(mkey))") : mkey
                df[!, mcol] = map(kv -> mfun(kv...), zip(df[!, :f], df[!, :f_ref]))
            end

            # remove reference spectra
            df = df[!, setdiff(propertynames(df), [:f_ref])]
        end
    end
    df = df[!, setdiff(propertynames(df), [:f])] # remove result spectra

    # store the metrics in file
    outfile = replace(spectrafile, "/spectra/" => "/metrics/")
    if eval_log_a
        outfile = replace(outfile, ".csv" => "_log_a.csv")
    end
    @info "Writing $(size(df, 1)) rows to $outfile.."
    CSV.write(outfile, df)
    return df
end

_metrics_classifier(df::DataFrame) =
    vcat(map( c -> begin
    
        @info "Let chi2s = $c.."
        @from i in df begin
            @where abs(i.chi2s) <= c # only consider methods that converged
            @group i by (i.name, i.skl, i.dataset, i.f_train, i.b) into g
            @let i_min = findmin(g.k)[2] # first k with chi2s <= c
            @select { name    = (g.name)[i_min],
                      skl     = (g.skl)[i_min],
                      dataset = (g.dataset)[i_min],
                      f_train = (g.f_train)[i_min],
                      b       = (g.b)[i_min],
                      k       = (g.k)[i_min],
                      chi2s   = c,
                      f       = (g.f)[i_min],
                      oob_acc = (g.oob_acc)[i_min],
                      ib_acc  = (g.ib_acc)[i_min],
                      oob_mse = (g.oob_mse)[i_min],
                      ib_mse  = (g.ib_mse)[i_min] }
            @collect DataFrame
        end
        
    end, 10.0 .^ -(1:6))...) # 1e-1, 1e-2, ..., 1e-6

_metrics_comparison(df::DataFrame) =
    vcat(map( c -> begin
    
        @info "Let chi2s = $c.."
        @from i in df begin
            @where abs(i.chi2s) <= c # only consider methods that converged
            @group i by (i.name, i.ac_regularisation, i.f_train, i.b) into g
            @let i_min = findmin(g.k)[2] # first k with chi2s <= c
            @select { name              = (g.name)[i_min],
                      method            = (g.method)[i_min],
                      fixweighting      = (g.fixweighting)[i_min],
                      stepsize          = (g.stepsize)[i_min],
                      smoothing         = (g.smoothing)[i_min],
                      discretization    = (g.discretization)[i_min],
                      n_df              = (g.n_df)[i_min],
                      J                 = (g.J)[i_min],
                      ac_regularisation = (g.ac_regularisation)[i_min],
                      f_train           = (g.f_train)[i_min],
                      b                 = (g.b)[i_min],
                      k                 = (g.k)[i_min],
                      chi2s             = c,
                      f                 = (g.f)[i_min] }
            @collect DataFrame
        end
        
    end, 10.0 .^ -(1:9))...) # 1e-1, 1e-2, ..., 1e-9

# columns to join reference spectra on (default is [ :b ])
JCOLS = Dict( "smearing"   => [ :b, :configfile ],
              "svd"        => [ :b, :dataset, :f_train ],
              "fit"        => [ :b, :dataset, :f_train ], # actually "fit_ratios" but the name is split at the first '_'
              "weightfix"  => [ :b, :dataset, :f_train, :skl ],
              "classifier" => [ :b, :dataset, :f_train ],
              "comparison" => [ :b, :f_train ],
              "feature" => [ :b, :dataset ] ) # actually "feature_selection"

# name of job that produced the result file (used to obtain the keys for JCOLS)
function _job_name(spectrafile::String)
    b = basename(spectrafile)
    return split(b, [ '_', '.' ])[startswith(b, "test_") ? 2 : 1]
end


"""
    aggregate_bootstrap(df, split_by, metric[; quantiles = (.05, .95), keep = Symbol[]])

Aggregate the `metric` from all bootstrap samples in the DataFrame `df`, which is split into
groups with the columns in `split_by`.

Returns the mean value and the lower and upper quantiles in another DataFrame with the
columns `y`, `y_plus`, and `y_minus`.

Keeps columns that are unique in each group, if they are specified in the `keeps` argument.
"""
aggregate_bootstrap(df::DataFrame, split_by::AbstractArray{Symbol,1}, metric::Symbol;
                    quantiles::Tuple{Float64,Float64}=(.05, .95), keep::Vector{Symbol}=Symbol[]) =
    combine(groupby(df, split_by)) do sdf
        m = sdf[isfinite.(sdf[!, metric]), metric] # metric array without NaNs and Infs
        m_mean = DataFrames.mean(m)
        m_lo, m_hi = DataFrames.quantile(m, [ quantiles... ]) # lower and upper quantiles
        cols = Pair{Symbol,Any}[
            :y => m_mean,
            :y_plus => m_hi - m_mean,
            :y_minus => m_mean - m_lo,
            :y_std => DataFrames.std(m; mean=m_mean)
        ]
        for col in keep
            if length(unique(sdf[!, col])) == 1
                push!(cols, col => sdf[1, col])
            else
                @warn "The value of $(col) is not unique" sdf[!, col]
            end # only keep unique values
        end
        DataFrame(; cols...) # aggregation of each group
    end


"""
    pdfpath(path, suffix="")

Obtain a file path to store a `.pdf` file in, which corresponds to the input file `path`.
"""
pdfpath(path::String, suffix::String="") = _outfilepath(path, "pdf", suffix)

"""
    texpath(path, suffix="")

Obtain a file path to store a `.tex` file in, which corresponds to the input file `path`.
"""
texpath(path::String, suffix::String="") = _outfilepath(path, "tex", suffix)

_outfilepath(path::String, ext::String, suffix::String) =
    replace(replace(path, "/metrics/" => "/$ext/"), r"(.*\.?.*)\..*" => s"\1") * "$suffix.$ext"


"""
    plot_histogram(bins, (f, f_style, f_name)...; kwargs...)

Plot histograms defined by their respective tuples `(f, f_style, f_name)` from histogram
values `f`, a PGFPlots style string `f_style`, and a legend entry `name`. The `bins` array
determines the positions on the x axis of each histogram.

**Keyword arguments**: `xlabel`, `ylabel`, `style`
"""
function plot_histogram( bins   :: AbstractArray{T1,1},
                         tups   :: Tuple{Array{T2,1},String,String}...;
                         xlabel :: String = "levels",
                         ylabel :: String = "edf",
                         style  :: String = "",
                         ymin   :: Float64 = 0.0 ) where {T1<:Number,T2<:Number}
    # add ending value to complete last line segment
    levelwidth = bins[2] - bins[1]
    bins = vcat(bins[1]-1e-12, bins, bins[end] + levelwidth, bins[end] + levelwidth+1e-12) # add last level
    
    # map tuples to plots
    plots = map(tups) do tup
        f, f_style, f_name = tup # split tuple
        f = vcat(ymin, f, f[end], ymin) # ending value
        Plots.Linear(bins, f, legendentry = f_name, style = joinstyles("histogram plot", f_style))
    end
    
    # return axis object
    return Axis([plots...]; style = joinstyles("histogram axis", "ymode = log, log origin y=infty", style),
                xmin = bins[1], xmax = bins[end], xlabel = xlabel, ylabel = ylabel)
end

"""
    progress_style(i)

Obtain the style number `i` for progress plots.
"""
progress_style(i::Int) = @sprintf "progress plot, m%02d" i

"""
    joinstyles(styles...)

Join the given PGFPlots style strings.
"""
function joinstyles(styles::String...)
    a = IOBuffer()
    for s in styles
        length(s) > 0 && print(a, a.size > 0 ? ", " : "", s)
    end
    String(take!(a))
end


# res implementations
include("res/gridsearch.jl")
include("res/svd.jl")
include("res/fit_ratios.jl")
include("res/ibu.jl")
include("res/data.jl")
include("res/subsampling.jl")
include("res/smearing.jl")
include("res/clustering.jl")
include("res/classifier.jl")
include("res/weightfix.jl")
include("res/stepsize.jl")
include("res/smoothing.jl")
include("res/expand_reduce.jl")
include("res/comparison.jl")
include("res/time_series_contributions.jl")
include("res/smart_control.jl")
include("res/feature_selection.jl")
include("res/uncertainty.jl")

include("res/amazon.jl")
include("res/roberta.jl")
include("res/dirichlet.jl")
include("res/main.jl")

end
