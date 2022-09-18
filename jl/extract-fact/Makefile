fact.csv: extract-fact.jl fact.hdf5 Manifest.toml
	julia --project="." $<
Manifest.toml: Project.toml
	julia --project="." --eval "using Pkg; Pkg.instantiate()"
fact.hdf5:
	curl --fail -o $@ https://factdata.app.tu-dortmund.de/dl2/FACT-Tools/v1.1.2/gamma_simulations_facttools_dl2.hdf5

.PRECIOUS: Manifest.toml fact.hdf5
