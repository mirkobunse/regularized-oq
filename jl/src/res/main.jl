METRICSFILES_MAIN = [
    "Books-OQ" => "res/metrics/amazon_roberta.csv",
    "FACT-OQ" => "res/metrics/dirichlet_fact.csv",
]
METRICSFILES_OTHERS = [
    "TFIDF" => "res/metrics/amazon_tfidf.csv",
    "blg-fdbck" => "res/metrics/dirichlet_blog-feedback.csv",
    "nln-nws" => "res/metrics/dirichlet_online-news-popularity.csv",
    "Ylnd" => "res/metrics/dirichlet_Yolanda.csv",
    "frd" => "res/metrics/dirichlet_fried.csv",
]
METHODS_MAIN = [
    :cc => ("MQ", "CC"), # first group: non-ordinal baselines
    :pcc => ("MQ", "PCC"),
    :acc => ("MQ", "ACC"),
    :pacc => ("MQ", "PACC"),
    :emq => ("MQ", "SLD"),
    :oqt => ("OQ", "OQT"), # second group: ordinal baselines
    :arc => ("OQ", "ARC"),
    :ibu => ("OQ", "IBU"),
    :prun => ("OQ", "RUN"),
    :semq => ("OQ+", "o-SLD"), # third group: new methods
    :nacc => ("OQ+", "o-ACC"),
    :npacc => ("OQ+", "o-PACC"),
]

"""
    main([outfile; metricsfiles])

Generate main plots from experimental results.
"""
function main(outfile="res/tex/main.tex"; metricsfiles=METRICSFILES_MAIN)
    df = vcat(map(metricsfiles) do (dataset, metricsfile)
        _df = coalesce.(CSV.read(metricsfile, DataFrame), "")
        _df[!, :dataset] .= dataset
        _df
    end...) # read all metricsfiles

    for (key, sdf) in pairs(groupby(df, :selection_metric))
        selection_metric = key.selection_metric # :nmd or :rnod
        agg_app = _main(
            df[df[!, :val_curvature_level].==-1, :],
            selection_metric
        )
        agg_app[!, :bin] .= "APP" # typical APP
        agg_1 = _main(
            df[(df.val_curvature_level.==1) .& (df.tst_curvature_level.==1), :],
            selection_metric
        )
        agg_1[!, :bin] .= "APP-1" # first bin
        agg = vcat(agg_app, agg_1)

        # unstack
        agg[!, :id] = agg[!, :method]
        agg[!, :variable] = agg[!, :dataset] .* " (" .* agg[!, :bin] .* ")"
        agg[!, :value] = "\$" .* _main_bold.(agg[!, :p_value]) .* "{" .* _main_avg.(agg[!, :avg]) .* " \\pm " .* _main_std.(agg[!, :std]) .* "}\$"
        agg = agg[!, [:id, :variable, :value]]
        tab = disallowmissing(coalesce.(unstack(agg), "---"))

        # sort the table according to the order of METHODS_MAIN
        methods = filter(x -> String(first(x)) ∈ agg[!, :id], METHODS_MAIN)
        sort!(tab, :id)
        tab[!, :order] = sortperm(first.(methods))
        sort!(tab, :order)
        tab[!, :group] = first.(last.(methods)) # group column
        tab[!, :method] = last.(last.(methods)) # method name column

        # write a LaTeX table to a text file
        _outfile = replace(outfile, ".tex" => "_$(selection_metric).tex")
        @info "Writing to $(_outfile)" selection_metric tab
        columns = vcat(
            [ "method" ],
            vcat(map(dataset -> [ "$(dataset) (APP)", "$(dataset) (APP-1)" ], first.(metricsfiles))...)
        )
        open(_outfile, "w") do io
            println(io, "\\begin{tabular}{l$(repeat("cc", length(metricsfiles)))}")
            println(io, "  \\toprule") # table header
            println(io, "  \\multirow{2}{*}{method} & $(join(map(d -> "\\multicolumn{2}{c}{$(d)}", first.(metricsfiles)), " & ")) \\\\")
            println(io, "  & $(join(map(d -> "APP & APP-20\\%", first.(metricsfiles)), " & ")) \\\\")
            print(io, "  \\midrule")
            last_group = tab[1, :group]
            for r in eachrow(tab) # write the table body row by row
                println(io, r[:group] == last_group ? "" : "[.5em]") # space between groups
                print(io, "  $(join([ r[c] for c in columns ], " & ")) \\\\")
                last_group = r[:group]
            end
            println(io, "\n  \\bottomrule")
            println(io, "\\end{tabular}")
        end
    end
end

_main_avg(x) = @sprintf("%.4f", x)[2:end] # ".nnnn"
_main_std(x) = @sprintf("%.3f", x)[2:end] # ".nnn"
_main_bold(x) = x >= 0.01 ? "\\mathbf" : ""

function _main(df::DataFrame, selection_metric::String)
    df = df[df[!, :selection_metric].==selection_metric, :]
    @info "_main receives a subset of $(nrow(df)) results"
    agg = combine(
        groupby(df, [:dataset, :method]),
        Symbol(selection_metric) => mean => :avg,
        Symbol(selection_metric) => std => :std
    ) # compute averages
    agg[!, :p_value] = map(eachrow(agg)) do row
        id = row[:method]
        ds = row[:dataset]
        _df = df[df[!, :dataset].==ds, :]
        _agg = agg[agg[!, :dataset].==ds, :]
        best_id = _agg[argmin(_agg[!, :avg]), :method]
        current_method = _df[(_df[!, :method].==id), :]
        best_method = _df[(_df[!, :method].==best_id), :]
        jdf = innerjoin(current_method, best_method;
            on = :sample,
            validate = (true, true), # check that the join column is a unique key
            makeunique = true, # rename left and right columns with renamecols
            renamecols = "_current" => "_best" # appendixes to the left and right column names
        )
        current_errors = jdf[!, Symbol("$(selection_metric)_current")]
        best_errors = jdf[!, Symbol("$(selection_metric)_best")]
        pvalue(SignedRankTest(current_errors, best_errors))
    end
    return agg
end
