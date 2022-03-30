# generate the ZIP archive from all tracked files
supplement.zip: $(filter-out supplement.zip .gitignore Makefile, $(shell git ls-files))
	zip -r $@ $^
