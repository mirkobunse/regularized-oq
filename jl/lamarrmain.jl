#
# julia -p <number of processes> main.jl [--no-validate] conf/gen/<file 1>.yml conf/gen/<file 2>.yml ...
#
using Distributed

# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
    using OrdinalQuantification
end

kwargs = Dict{Symbol,Any}()
if "--no-validate" ∈ ARGS
    kwargs[:validate] = false
end
files = filter(x -> x != "--no-validate", ARGS)
@info "Received $(length(files)) jobs:\n\t" * join(files, "\n\t")
for file in files
    if !endswith(file, ".yml")
        @warn "Skipping $file, which does not end on '.yml'"
    elseif !isfile(file)
        @warn "Skipping $file, which is not a file"
    else
        Experiments.run(file; kwargs...) # start the current configuration
    end
end
