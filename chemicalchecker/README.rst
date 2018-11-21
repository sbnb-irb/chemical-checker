===============
chemicalchecker
===============

Python Package with everything needed to create and query the Chemical Checker.


* Free software: GNU General Public License v3
* Documentation: http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/wikis/home


Features
--------

* TODO


Unit-testing
----

Got to your chemicalchecker package directory

To run all tests::

$ pytest

To run a subset of tests::

$ pytest tests.test_chemicalchecker


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

