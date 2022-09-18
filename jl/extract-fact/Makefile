fact.csv: extract-fact.jl Manifest.toml
	julia --project="." $<
Manifest.toml:
	julia --project="." --eval "using Pkg; Pkg.instantiate()"

.PRECIOUS: Manifest.toml fact.hdf5
