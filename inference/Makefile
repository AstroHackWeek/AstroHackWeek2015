SLIDES = probability_slides
QUESTIONS = probability_questions
NS = nested_sampling

TEXFILES = *.tex

all: $(SLIDES) $(QUESTIONS) $(NS)

clean:
	-rm -f *.aux *.log *.dvi *.blg *.bbl *.snm *.nav *.toc *.out
	# -rm -f  $(SLIDES).pdf $(QUESTIONS).pdf  $(NS).pdf

pdf: $(PAPER)

$(SLIDES): $(SLIDES).tex
	pdflatex -shell-escape $(SLIDES).tex
	pdflatex -shell-escape $(SLIDES).tex

$(QUESTIONS): $(QUESTIONS).tex
	pdflatex $(QUESTIONS).tex
	pdflatex $(QUESTIONS).tex

$(NS): $(NS).tex
	pdflatex --shell-escape $(NS).tex
	pdflatex --shell-escape $(NS).tex
