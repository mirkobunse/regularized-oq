#
# julia -p <number of processes> main.jl conf/gen/<file 1>.yml conf/gen/<file 2>.yml ...
#
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
        Experiments.run(arg) # start the current configuration
    end
end
