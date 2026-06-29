# presentation/paper/Makefile — build the variant-cache writeup.
#
#   make            # check figures, then build main.pdf
#   make pdf        # build main.pdf (latexmk -> pdflatex)
#   make check      # assert every referenced figure exists
#   make clean      # remove latex aux files
#   make distclean  # also remove main.pdf
#
# Requires a TeX distribution (latexmk + pdflatex). The 13_vc_vs_fullforward.png
# and a regenerated 05_ar_curves.png come from the GPU steps; until then the
# doc builds against whatever PNGs are present (see `make check`).

PYTHON ?= /home/andrew.dickson/.conda/envs/svar/bin/python
MAIN   := main

.PHONY: all pdf check figs clean distclean

all: check pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Refresh the bundled figures from the source presentation/figs/ dir.
figs:
	cp ../figs/*.png figs/

check:
	$(PYTHON) check_figures.py

clean:
	latexmk -c $(MAIN).tex 2>/dev/null || true
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).fls $(MAIN).fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf
