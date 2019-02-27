# generate api rst from code docstrings
# -f overwrite existing
# -e each module on it's own page
sphinx-apidoc -o source ../chemicalchecker -f -e
# generate doc html in docs/_build/html
make html
