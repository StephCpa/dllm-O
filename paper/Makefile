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

# Figure 1 and the optional Figure 1(b) candidate are manually reviewed assets
# drawn by their own scripts in figures/; distclean removes only build outputs.
GENERATED_FIGURES := fig2_block_size fig3_policy_frontier fig4_diversity fig5_temperature_frontier

distclean:
	$(LATEXMK) -C main.tex
	rm -f $(foreach f,$(GENERATED_FIGURES),figures/$(f).pdf figures/$(f).png) generated/*.tex
