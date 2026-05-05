MAIN=awareness_stratified_em_brief

.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
