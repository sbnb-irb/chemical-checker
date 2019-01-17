#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['six', 'autologging', 'paramiko', 'termcolor', 'standardiser',
                'psycopg2', 'xlrd', 'cmapPy', 'e3fp', 'gensim', 'csvsort',
                'intbitset', 'cython', 'sqlalchemy', 'patool', 'wget',
                'timeout_decorator', 'numpy', 'pandas', 'scipy', 'theano',
                'h5py', 'tqdm', 'networkx', 'matplotlib', 'seaborn',
                'scikit-learn', 'tensorflow', 'adanet', 'keras', 'hdbscan',
                'MulticoreTSNE', 'fancyimpute', 'pybel']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="SBNB",
    author_email='sbnb@irbbarcelona.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python Package with everything needed to create and query the Chemical Checker.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='chemicalchecker',
    name='chemicalchecker',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='http://gitlab.sbnb.org/project-specific-repositories/chemical_checker',
    version='0.1.0',
    zip_safe=False,
)
