fact.csv: extract-fact.jl fact.hdf5 Manifest.toml
	julia --project="." $<
Manifest.toml: Project.toml
	julia --project="." --eval "using Pkg; Pkg.instantiate()"
fact.hdf5:
	curl --fail -o $@ https://factdata.app.tu-dortmund.de/dl2/FACT-Tools/v1.1.2/gamma_simulations_facttools_dl2.hdf5

# what we have uploaded
extract-fact-oq.zip: extract-fact.jl Makefile Manifest.toml Project.toml README.md ../app_val_indices.csv ../app-oq_val_indices.csv ../app_tst_indices.csv ../app-oq_tst_indices.csv
	zip $@ $^

.PRECIOUS: Manifest.toml fact.hdf5
