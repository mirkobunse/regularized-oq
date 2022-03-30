using Pkg
uuid = "4006ddf5-3342-4a8e-96e9-cd0df3b68585" # UUID of DeconvExp
if !haskey(Pkg.dependencies(), Base.UUID(uuid))
    Pkg.develop(PackageSpec(pwd()))
end
