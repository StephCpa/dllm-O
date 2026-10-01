PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all assets clean distclean

all: main.pdf

assets:
	$(PYTHON) scripts/generate_manuscript_assets.py

main.pdf: main.tex references.bib assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -c main.tex

distclean:
	$(LATEXMK) -C main.tex
	rm -f figures/*.pdf figures/*.png generated/paper_numbers.tex
