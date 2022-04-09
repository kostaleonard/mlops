all: help

help:
	@echo "To install required packages, run 'make install' from a clean 'python:3.9' (or higher) conda environment."

install:
	pip install -r requirements.txt

lint:
	pylint mlops
	pylint tests

test:
	pytest -m "not slowtest" --cov=mlops tests
	coverage xml

test_slow:
	pytest -m "slowtest" --cov=mlops tests
	coverage xml

test_mocked_aws:
	pytest -m "mockedawstest" --cov=mlops tests
	coverage xml

test_full:
	pytest --cov=mlops tests
	coverage xml

documentation:
	cd docs && make clean
	rm -rf docs/_apidoc
	cd docs && sphinx-apidoc -o _apidoc ../mlops
	cd docs && make html

package_prod:
	rm -rf dist
	python3 -m build
	python3 -m twine upload dist/*

package_test:
	rm -rf dist
	python3 -m build
	python3 -m twine upload --repository testpypi dist/*
