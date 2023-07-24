online-news-popularity.csv: extract-oq.jl Manifest.toml
	julia --project="." $<
Manifest.toml: Project.toml
	julia --project="." --eval "using Pkg; Pkg.instantiate()"

# what we have uploaded
extract-oq.zip: extract-oq.jl openml.yml uci.yml Makefile Manifest.toml Project.toml README.md blog-feedback_app_val_indices.csv blog-feedback_app-oq_val_indices.csv blog-feedback_app_tst_indices.csv blog-feedback_app-oq_tst_indices.csv fried_app_val_indices.csv fried_app-oq_val_indices.csv fried_app_tst_indices.csv fried_app-oq_tst_indices.csv online-news-popularity_app_val_indices.csv online-news-popularity_app-oq_val_indices.csv online-news-popularity_app_tst_indices.csv online-news-popularity_app-oq_tst_indices.csv Yolanda_app_val_indices.csv Yolanda_app-oq_val_indices.csv Yolanda_app_tst_indices.csv Yolanda_app-oq_tst_indices.csv
	zip --junk-paths $@ $^

.PRECIOUS: Manifest.toml
