# generate a ZIP archive from all tracked files
CONTENT=$(filter-out CHECK.txt TEST_CHECK.txt %.gitignore Makefile, $(shell git ls-files)) supplement.pdf
supplement.zip: $(CONTENT)
	rm -f $@ && zip -r $@ $^

supplement.pdf: supplement.tex $(wildcard jl/res/tex/*)
	- pdflatex $<
	pdflatex $<

# check for phrases that might reveal our identities
check: CHECK.txt
	egrep --color="auto" -i -f $< $(CONTENT)
.PHONY: check
