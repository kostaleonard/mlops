all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

pylint:
	pylint mlops
	pylint tests

pytest:
	pytest -m "not slowtest and not awstest" --cov=mlops tests
	coverage xml

pytest_slow:
	pytest -m "slowtest" --cov=mlops tests
	coverage xml

pytest_aws:
	pytest -m "awstest" --cov=mlops tests
	coverage xml

pytest_full:
	pytest --cov=mlops tests
	coverage xml

documentation:
	# TODO see adjutant

package_prod:
	rm -rf dist
	python3 -m build
	python3 -m twine upload dist/*

package_test:
	rm -rf dist
	python3 -m build
	python3 -m twine upload --repository testpypi dist/*
