using Pkg
uuid = "db81655c-0675-4b96-98de-d8246fe7af2c" # UUID of OrdinalQuantification.jl
if !haskey(Pkg.dependencies(), Base.UUID(uuid))
    Pkg.develop(PackageSpec(pwd()))
end
