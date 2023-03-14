module Results

using
    CherenkovDeconvolution,
    CSV,
    DataFrames,
    HypothesisTests,
    OrderedCollections,
    Printf,
    Statistics

METRICSFILES_MAIN = [
    "\\textsc{Amazon-OQ-BK}" => "res/csv/amazon_roberta.csv",
    "\\textsc{Fact-OQ}" => "res/csv/dirichlet_fact.csv",
]
METRICSFILES_TFIDF = [
    "\\textsc{Amazon-OQ-BK} (TFIDF)" => "res/csv/amazon_tfidf.csv",
]
METRICSFILES_OTHERS = [
    "\\textsc{Uci-blog-feedback-OQ}" => "res/csv/dirichlet_blog-feedback.csv",
    "\\textsc{Uci-online-news-popularity-OQ}" => "res/csv/dirichlet_online-news-popularity.csv",
    "\\textsc{OpenMl-Yolanda-OQ}" => "res/csv/dirichlet_Yolanda.csv",
    "\\textsc{OpenMl-fried-OQ}" => "res/csv/dirichlet_fried.csv",
]
METHODS_MAIN = [
    :cc => ("MQ", "CC"), # first group: non-ordinal baselines
    :pcc => ("MQ", "PCC"),
    :acc => ("MQ", "ACC"),
    :pacc => ("MQ", "PACC"),
    :hdx => ("MQ", "HDx"),
    :hdy => ("MQ", "HDy"),
    :sld => ("MQ", "SLD"),
    :oqt => ("OQ", "OQT"), # second group: ordinal baselines
    :arc => ("OQ", "ARC"),
    :ibu => ("OQ", "IBU"),
    :run => ("OQ", "RUN"),
    :svd => ("OQ", "SVD"),
    :oacc => ("OQ+", "o-ACC"), # third group: new methods
    :opacc => ("OQ+", "o-PACC"),
    :ohdx => ("OQ+", "o-HDx"),
    :ohdy => ("OQ+", "o-HDy"),
    :osld => ("OQ+", "o-SLD"),
]

main_tfidf() = Results.main("res/tex/main_tfidf.tex"; metricsfiles=Results.METRICSFILES_TFIDF)
main_others() = Results.main("res/tex/main_others.tex"; metricsfiles=Results.METRICSFILES_OTHERS)

"""
    main([outfile; metricsfiles])
    main_tfidf()
    main_others()

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
            sdf[sdf[!, :val_curvature_level].==-1, :],
            selection_metric
        )
        agg_app[!, :bin] .= "APP" # typical APP
        agg_1 = _main(
            sdf[(sdf.val_curvature_level.==1) .& (sdf.tst_curvature_level.==1), :],
            selection_metric
        )
        agg_1[!, :bin] .= "APP-1" # first bin
        agg = vcat(agg_app, agg_1)

        # unstack
        agg[!, :id] = agg[!, :validation_group]
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
        tab[!, :validation_group] = last.(last.(methods)) # method name column

        # write a LaTeX table to a text file
        _outfile = replace(outfile, ".tex" => "_$(selection_metric).tex")
        @info "Writing to $(_outfile)" selection_metric tab
        columns = vcat(
            [ "validation_group" ],
            vcat(map(dataset -> [ "$(dataset) (APP)", "$(dataset) (APP-1)" ], first.(metricsfiles))...)
        )
        open(_outfile, "w") do io
            println(io, "\\begin{tabular}{l$(repeat("cc", length(metricsfiles)))}")
            println(io, "  \\toprule") # table header
            println(io, "  \\multirow{2}{*}{method} & $(join(map(d -> "\\multicolumn{2}{c}{$(d)}", first.(metricsfiles)), " & ")) \\\\")
            println(io, "  & $(join(map(d -> "APP & APP-OQ", first.(metricsfiles)), " & ")) \\\\")
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

function _main(df::DataFrame, selection_metric::AbstractString)
    @info "_main receives a subset of $(nrow(df)) results"
    agg = combine(
        groupby(df, [:dataset, :validation_group]),
        Symbol(selection_metric) => mean => :avg,
        Symbol(selection_metric) => std => :std
    ) # compute averages
    agg[!, :p_value] = map(eachrow(agg)) do row
        id = row[:validation_group]
        ds = row[:dataset]
        _df = df[df[!, :dataset].==ds, :]
        _agg = agg[agg[!, :dataset].==ds, :]
        best_id = _agg[argmin(_agg[!, :avg]), :validation_group]
        current_method = _df[(_df[!, :validation_group].==id), :]
        best_method = _df[(_df[!, :validation_group].==best_id), :]
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

"""
    ranking(metricsfile)

Generate ranking tables from the results obtained with a single data set.
"""
function ranking(metricsfile::String)
    df = coalesce.(CSV.read(metricsfile, DataFrame), "") # read the metricsfile
    regex = r"(.+)/csv/(.+)\.csv" # what to replace in the file name
    if :val_curvature_level ∉ propertynames(df)
        @warn "Adding missing curvature levels; are these validations results?"
        df[!,:val_curvature_level] .= -1
        df[!,:tst_curvature_level] = df[!,:curvature_level]
        df[!,:selection_metric] .= "nmd"
    end

    for (key, sdf) in pairs(groupby(df, :selection_metric))
        metric = key.selection_metric # :nmd or :rnod

        # one output for typical APP, validating and testing on all samples
        _ranking_curvature(
            sdf[sdf[!, :val_curvature_level].==-1, :],
            replace(
                metricsfile,
                regex => SubstitutionString("\\1/tex/\\2_$(metric)_all.tex")
            ), # output path: replace with "*/tex/*_<metric name>_all.tex"
            Symbol(metric) => uppercase(metric) # pair metric_column => metric_name
        )

        # one additional output for each curvature level, used both for validation and testing
        levels = sort(unique(sdf[!, :tst_curvature_level]))
        for level in levels
            _ranking_curvature(
                sdf[(sdf.val_curvature_level.==level) .& (sdf.tst_curvature_level.==level), :],
                replace(
                    metricsfile,
                    regex => SubstitutionString("\\1/tex/\\2_$(metric)_$(level).tex")
                ), # replace with "*/tex/*_<metric name>_<curvature level>.tex"
                Symbol(metric) => uppercase(metric)
            )
        end
    end
end

function _ranking_curvature(df::DataFrame, outfile::String, metric::Pair{Symbol,String})
    if nrow(df) == 0
        @warn "Skipping $(outfile) for having no results; are these validations results?"
        return
    end
    @debug "Receiving a subset of $(nrow(df)) results for $(outfile)"
    if outfile[end-3:end] != ".tex"
        error("$(outfile) does not end on '.tex'; please use inputs from res/csv/")
    end

    # compute averages to sort by
    agg = sort(combine(
        groupby(df, [:name, :validation_group]),
        metric[1] => mean => :avg,
        metric[1] => std => :std
    ), :avg)
    agg = sort(combine(
        groupby(agg, :validation_group),
        sdf -> sort(sdf, :avg)[1,:] # select the best
    ), :avg)
    agg[!, :p_value] = map(agg[!, :name]) do name
        jdf = innerjoin(
            df[df[!, :name] .== name, [:sample, metric[1]]], # the current method
            df[df[!, :name] .== agg[1, :name], [:sample, metric[1]]]; # the best method
            on = :sample,
            validate = (true, true), # check that the :sample column is a unique key
            makeunique = true, # rename left and right columns with renamecols
            renamecols = "_this" => "_best" # appendixes to the left and right column names
        )
        this_metric = jdf[!, Symbol("$(metric[1])_this")]
        best_metric = jdf[!, Symbol("$(metric[1])_best")]
        pvalue(SignedRankTest(this_metric, best_metric))
    end

    # write a LaTeX table to a text file
    @info "Writing to $outfile"
    open(outfile, "w") do io
        println(io, "\\begin{tabular}{lc}")
        println(io, "  \\toprule")
        println(io,
            "  quantification method & ",
            "avg. ",
            metric[2],
            " \$\\pm\$ stddev. \\\\"
        ) # table header
        println(io, "  \\midrule")
        for r in eachrow(agg) # write the table body row by row
            print(io, "  ", r[:name], " & \$") # print with indentation
            if r[:p_value] >= 0.01
                print(io, "\\mathbf{")
            end
            print(io, @sprintf("%.4f", r[:avg]), " \\pm ", @sprintf("%.4f", r[:std]))
            if r[:p_value] >= 0.01
                print(io, "}")
            end
            println(io, "\$ \\\\")
        end
        println(io, "  \\bottomrule")
        println(io, "\\end{tabular}")
    end
end

CASTANO_ROWS = [ # specify the order
    "SWD",
    "ESL",
    "LEV",
    "cement_strength_gago",
    "stock.ord",
    "auto.data.ord_chu",
    "bostonhousing.ord_chu",
    "californiahousing_gago",
    "winequality-red_gago",
    "winequality-white_gago_rev",
    "skill_gago",
    "SkillCraft1_rev_7clases",
    "kinematics_gago",
    "SkillCraft1_rev_8clases",
    "ERA",
    "ailerons_gago",
    "abalone.ord_chu",
]

"""
    castano(means_file)

Generate tables from the `means_file` output of the Castano experiment. Typically, this file is called `res/castano/yyyy-mm-dd_hh:mm:ss/means_CV(DECOMP)_10x300CV20_170.csv`.
"""
function castano(means_file::String)
    df = coalesce.(CSV.read(means_file, DataFrame), "") # read the means_file
    for (error, argbest) in [ "emd_score" => argmax ] # extend with "nmd"=>argmin
        df_error = df[
            df[!, :error] .== error, # select rows
            setdiff(names(df), ["decomposer", "error"]) # remove constant columns
        ]
        df_error = df_error[sortperm(df_error[!,:dataset])[invperm(sortperm(CASTANO_ROWS))],:]
        push!(df_error, vcat( # compute average performance
            ["average"],
            mean.(eachcol(df_error[!, names(df_error)[2:end]]))
        ))

        # select the best method / column per group
        method_groups = [ match(r"^([\w-]+)", n)[1] for n in names(df_error) ]
        select_best = g -> begin
            r = df_error[end, method_groups .== g] # DataFrame row
            if length(r) > 1 # log information about parameter outcomes
                r_df = DataFrame(r)
                r_df[!, :method] = [:value]
                @info g sort(permutedims(r_df, :method), :value)
            end
            argbest(r) # take out the actual selection
        end
        df_error = df_error[!, [ select_best(g) for g in unique(method_groups) ]]
        rename!(df_error, [ match(r"^([\w-]+)", n)[1] for n in names(df_error) ])

        # write a LaTeX table to a text file
        outfile = splitext(means_file)[1] * "_" * error * ".tex"
        @info "Writing to $outfile"
        open(outfile, "w") do io
            println(io, "\\begin{tabular}{l$("c"^(length(names(df_error))-1))}")
            println(io, "  \\toprule")
            println(io, "  " * join(names(df_error), " & ") * " \\\\") # table header
            println(io, "  \\midrule")
            for r in eachrow(df_error) # write the table body row by row
                best_method = argbest(r[2:end]) # <: Symbol
                contents = OrderedDict{Symbol,String}(
                    :dataset => replace(r[:dataset], "_" => "\\_")
                )
                for (k, v) in zip(keys(r[2:end]), values(r[2:end]))
                    contents[k] = (@sprintf "%.4f" v)[2:end]
                end
                contents[best_method] = "\\textbf{$(contents[best_method])}"
                println(io, "  " * join(values(contents), " & ") * " \\\\")
            end
            println(io, "  \\bottomrule")
            println(io, "\\end{tabular}")
        end
    end
end

end
