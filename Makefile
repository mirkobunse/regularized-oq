# generate the ZIP archive from all tracked files
supplement.zip: $(filter-out CHECK.txt %.gitignore Makefile, $(shell git ls-files))
	zip -r $@ $^

# check for phrases that might reveal our identities
check: CHECK.txt
	grep --color="auto" -i -f $< $(filter-out %.gitignore Makefile, $(shell git ls-files))
.PHONY: check
