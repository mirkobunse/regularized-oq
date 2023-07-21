#
# julia extract-app-oq.jl
#
using CherenkovDeconvolution.DeconvUtil, DataFrames, LinearAlgebra, OrdinalQuantification

function f_true(input_path::String)
    _, y_tst = Experiments.load_amazon_data(input_path)
    return DeconvUtil.fit_pdf(y_tst, 0:4)
end

function sample_curvature(f_true::Vector{Float64})
    C = LinearAlgebra.diagm(
        -1 => fill(-1, 4),
        0 => fill(2, 5),
        1 => fill(-1, 4)
    )[2:4, :]
    return sum((C*f_true).^2)
end

for (input_dir, output_dir) ∈ [
        "/mnt/data/amazon-oq-bk/roberta/app/dev_samples" => "/mnt/data/amazon-oq-bk/roberta/app-oq/dev_samples",
        "/mnt/data/amazon-oq-bk/roberta/app/test_samples" => "/mnt/data/amazon-oq-bk/roberta/app-oq/test_samples",
        "/mnt/data/amazon-oq-bk/tfidf/app/dev_samples" => "/mnt/data/amazon-oq-bk/tfidf/app-oq/dev_samples",
        "/mnt/data/amazon-oq-bk/tfidf/app/test_samples" => "/mnt/data/amazon-oq-bk/tfidf/app-oq/test_samples",
        ]
    num_samples = length(readdir(input_dir))
    df = DataFrame(input_path = [ input_dir * "/$(i-1).txt" for i ∈ 1:num_samples ])

    # also read the ground-truth prevalences
    prevalences = readlines(replace(input_dir, "_samples" => "_prevalences.txt"))
    header = prevalences[1] # first line
    df[!, :prevalences] = prevalences[2:end]

    # compute the smoothness of each sample
    df[!, :f_true] = map(f_true, df[!, :input_path])
    df[!, :sample_curvature] = map(sample_curvature, df[!, :f_true])

    # select the smoothest 20% of all samples
    df[!, :curvature_level] = Experiments._curvature_level(df, 5) # 5 splits ≡ 20% per split
    df = df[df[!, :curvature_level] .== 1, :] # filter

    # replace IDs in prevalences lines
    df[!, :prevalences] = [ join(vcat(x, split(y, ",")[2:end]), ",") for (x, y) ∈ enumerate(df[!, :prevalences]) ]

    # define output paths and copy files
    df[!, :output_path] = [ output_dir * "/$(i-1).txt" for i ∈ 1:nrow(df) ]
    mkpath(output_dir)
    for r in eachrow(df)
        try
            cp(r[:input_path], r[:output_path])
        catch any_exception
            if !isa(any_exception, ArgumentError)
                rethrow()
            end # ignore ArgumentErrors
        end
    end

    # also write a new prevalences file
    open(replace(output_dir, "_samples" => "_prevalences.txt"), "w") do f
        println(f, header)
        for l ∈ df[!, :prevalences]
            println(f, l)
        end
    end

    @info "$(output_dir) succesfully created with $(length(readdir(output_dir))) files" df[1:3, :input_path] df[1:3, :prevalences] df[1:3, :f_true]
end
