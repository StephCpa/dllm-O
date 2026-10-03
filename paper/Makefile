PYTHON ?= python3
LATEXMK ?= latexmk

# A fixed build date (2026-10-01T00:00:00Z) makes pdfTeX write the same
# /CreationDate, /ModDate and /ID on every build, so main.pdf is
# byte-identical across rebuilds with the same TeX distribution.
export SOURCE_DATE_EPOCH ?= 1790812800
export FORCE_SOURCE_DATE = 1

.PHONY: all assets figure1 manifest check clean distclean

all: main.pdf

assets:
	$(PYTHON) scripts/generate_manuscript_assets.py

main.pdf: main.tex references.bib assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

# Figure 1 is a manually reviewed schematic and is not redrawn by `make`.
figure1:
	$(PYTHON) figures/fig1_overview.py

# Record the hashes of the current build outputs after an intended change.
manifest:
	$(PYTHON) scripts/build_manifest.py --write

# Rebuild everything from scratch and compare with build_manifest.json.
check: distclean
	$(PYTHON) figures/fig1_overview.py
	$(MAKE) all
	$(PYTHON) scripts/build_manifest.py --check

clean:
	$(LATEXMK) -c main.tex

# Figure 1 and the optional Figure 1(b) candidate are manually reviewed assets
# drawn by their own scripts in figures/; distclean removes only build outputs.
GENERATED_FIGURES := fig2_block_size fig3_policy_frontier fig4_diversity fig5_temperature_frontier

distclean:
	$(LATEXMK) -C main.tex
	rm -f $(foreach f,$(GENERATED_FIGURES),figures/$(f).pdf figures/$(f).png) generated/*.tex
