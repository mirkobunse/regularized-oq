"""
    amazon(metricsfile="res/metrics/amazon.csv")

Generate amazon plots from experimental results.
"""
function amazon(metricsfile::String="res/metrics/amazon.csv"; __initialization::String="amazon")
    initialize(__initialization, metricsfile) # initialize PGFPlots
    df = coalesce.(CSV.read(metricsfile, DataFrame), "") # read the metricsfile
    regex = r"(.+)/metrics/(.+)\.csv" # what to replace in the file name

    for (key, sdf) in pairs(groupby(df, :selection_metric))
        metric = key.selection_metric # :nmd or :rnod

        # one output for typical APP, validating and testing on all samples
        _amazon_curvature(
            sdf[sdf[!, :val_curvature_level].==-1, :],
            replace(
                metricsfile,
                regex => SubstitutionString("\\1/table-tex/\\2_$(metric)_all.tex")
            ), # output path: replace with "*/table-tex/*_<metric name>_all.tex"
            Symbol(metric) => uppercase(metric) # pair metric_column => metric_name
        )

        # one additional output for each curvature level, used both for validation and testing
        levels = sort(unique(sdf[!, :tst_curvature_level]))
        for level in levels
            _amazon_curvature(
                sdf[(sdf.val_curvature_level.==level) .& (sdf.tst_curvature_level.==level), :],
                replace(
                    metricsfile,
                    regex => SubstitutionString("\\1/table-tex/\\2_$(metric)_$(level).tex")
                ), # replace with "*/table-tex/*_<metric name>_<curvature level>.tex"
                Symbol(metric) => uppercase(metric)
            )
        end

        # generate a Critical Difference Diagram
        pairs_app = CriticalDifferenceDiagrams._to_pairs(
                sdf[sdf[!, :val_curvature_level].==-1, :],
                :method, # "treatment" column
                :sample, # "observation" column
                Symbol(metric) # "outcome" column
        )
        sort!(pairs_app, by=first) # sort pairs by their first element
        sequence = [ "regular APP" => pairs_app ] # a sequence of CD diagrams
        for level in levels # add one more element for each level
            pairs_level = CriticalDifferenceDiagrams._to_pairs(
                sdf[(sdf.val_curvature_level.==level) .& (sdf.tst_curvature_level.==level), :],
                :method, :sample, Symbol(metric)
            )
            sort!(pairs_level, by=first) # sort pairs by their first element
            push!(sequence, "APP bucket $(level)" => pairs_level)
        end
        plot = CriticalDifferenceDiagrams.plot(sequence...; maximize_outcome=false, alpha=0.01)
        outfile = replace(
            metricsfile,
            regex => SubstitutionString("\\1/table-tex/\\2_$(metric)_cdd.tex")
        )
        @info "Writing to $outfile"
        PGFPlots.save(outfile, plot; limit_to=:picture)
    end
end

function _amazon_curvature(df::DataFrame, outfile::String, metric::Pair{Symbol,String})
    @info "Receiving a subset of $(nrow(df)) results for $(outfile)"

    # compute averages to sort by
    agg = sort(combine(
        groupby(df, [:name, :method]),
        metric[1] => mean => :avg,
        metric[1] => std => :std
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
        # @info "Foo" name best=agg[1, :name] jdf
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
