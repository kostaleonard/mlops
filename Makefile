all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

pylint:
	pylint mlops
	pylint tests

pytest:
	pytest -m "not slowtest" --cov=mlops tests

pytest_include_slow:
	pytest --cov=mlops tests

documentation:
	# TODO see adjutant

package_prod:
	# TODO see adjutant

package_test:
	# TODO see adjutant
