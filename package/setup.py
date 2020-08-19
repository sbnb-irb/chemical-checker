#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    'numpy',
    'h5py',
    'psycopg2-binary',
    'pandas',
    'networkx',
    'autologging',
    'scipy',
    'sqlalchemy',
    'paramiko',
    'sklearn',
    'csvsort',
    'matplotlib<3.0',
    'seaborn',
    'tqdm',
    'apache-airflow'
]

setup_requirements = ['pytest-runner']

test_requirements = [
    'pytest',
    'mock'
]


setup(
    author="SBNB",
    author_email='sbnb@irbbarcelona.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Chemical Checker Package.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description='''The Chemical Checker (CC) is a data-driven resource of small molecule
bioactivity data. The main goal of the CC is to express data in a format
that can be used off-the-shelf in daily computational drug discovery
tasks. The resource is organized in **5 levels** of increasing
complexity, ranging from the chemical properties of the compounds to
their clinical outcomes. In between, we consider targets, off-targets,
perturbed biological networks and several cell-based assays, including
gene expression, growth inhibition, and morphological profiles. The CC
is different to other integrative compounds database in almost every
aspect. The classical, relational representation of the data is
surpassed here by a less explicit, more machine-learning-friendly
abstraction of the data''',
    include_package_data=True,
    keywords='chemicalchecker',
    name='chemicalchecker',
    packages=find_packages(
        exclude=['chemicalchecker.tool.hotnet*', 'chemicalchecker.tool.targetmate*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='http://gitlab.sbnb.org/packages/chemical_checker',
    version='0.2.2',
    zip_safe=False,
)
