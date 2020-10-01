from setuptools import setup, find_packages

__author__ = """SBNB"""
__email__ = 'sbnb@irbbarcelona.org'
__version__ = '1.0.0'

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
]

setup_requirements = ['pytest-runner']

test_requirements = [
    'pytest',
    'mock'
]

setup(
    name='chemicalchecker',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="Chemical Checker Package.",
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
    url='http://gitlab.sbnb.org/packages/chemical_checker',
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite='tests',
    zip_safe=False,
    include_package_data=True,
    license="MIT License",
    keywords='chemicalchecker bioactivity signatures chemoinformatics',
    classifiers=[
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
    ]
)
