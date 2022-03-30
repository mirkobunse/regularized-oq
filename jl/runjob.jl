@info "using DeconvExp"
using DeconvExp

@info "using @everywhere"
using Distributed: @everywhere
@everywhere using ComfyCommons.ComfyLogging; set_global_logger()
@everywhere using DeconvExp

@info "Received $(length(ARGS)) job(s):\n\t" * join(ARGS, "\n\t")
for arg in ARGS
    if !endswith(arg, ".yml")
        @warn "Skipping $arg, which does not end on '.yml'"
    elseif !isfile(arg)
        @warn "Skipping $arg, which is not a file"
    else
        DeconvExp.Job.run(arg) # start the current configuration
    end
end
