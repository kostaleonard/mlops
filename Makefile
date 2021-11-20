all: help

help:
	# TODO

install:
	pip install -r requirements.txt

pylint:
	pylint mlops
	pylint tests
