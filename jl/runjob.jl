@info "TODO move runjob.jl to src/Job.jl"
using OrdinalQuantification

@info "using @everywhere"
using Distributed: @everywhere
@everywhere using OrdinalQuantification

@info "Received $(length(ARGS)) job(s):\n\t" * join(ARGS, "\n\t")
for arg in ARGS
    if !endswith(arg, ".yml")
        @warn "Skipping $arg, which does not end on '.yml'"
    elseif !isfile(arg)
        @warn "Skipping $arg, which is not a file"
    else
        OrdinalQuantification.Job.run(arg) # start the current configuration
    end
end
