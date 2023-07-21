online-news-popularity.csv: extract-others.jl Manifest.toml
	julia --project="." $<
Manifest.toml: Project.toml
	julia --project="." --eval "using Pkg; Pkg.instantiate()"

# what we have uploaded
extract-others.zip: extract-others.jl Makefile Manifest.toml Project.toml README.md ../app_val_indices.csv ../app-oq_val_indices.csv ../app_tst_indices.csv ../app-oq_tst_indices.csv
	zip --junk-paths $@ $^

.PRECIOUS: Manifest.toml
