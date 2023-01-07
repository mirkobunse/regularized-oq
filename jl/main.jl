#
# julia -p <number of processes> main.jl [--no-validate] conf/gen/<file 1>.yml conf/gen/<file 2>.yml ...
#
using OrdinalQuantification

@info "using @everywhere"
using Distributed: @everywhere
@everywhere using OrdinalQuantification

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
