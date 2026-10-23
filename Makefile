PYTHON ?= python3

.PHONY: demo test test-unittest

demo:
	$(PYTHON) -m stocka demo

test:
	$(PYTHON) -m pytest -q

test-unittest:
	$(PYTHON) -m unittest discover -s brain/tests -t . -q
	$(PYTHON) -m unittest discover -s tests -t tests -q
