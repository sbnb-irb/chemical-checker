.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 chemicalchecker tests

test: ## run tests quickly with the default Python
	pytest tests/ -s

docs: ## generate Sphinx HTML documentation, including API docs
	rm -fr docs/_build/
	rm -fr docs/_autosummary/
	##sphinx-apidoc -o docs/source chemicalchecker -f -M -e ## updating sphinx made this useless
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload --verbose dist/*

release-test: dist ## package and upload a release to test server
	twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel

install: clean ## install the package to the active Python's site-packages
	python setup.py install
