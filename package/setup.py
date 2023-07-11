from setuptools import setup, find_packages

__author__ = """SBNB"""
__email__ = 'sbnb@irbbarcelona.org'
__version__ = '1.0.3'

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
    'scikit-learn',
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
    long_description=open('README.rst').read().strip(),
    long_description_content_type='text/x-rst',
    url='http://gitlabsbnb.irbbarcelona.org/packages/chemical_checker',
    packages=find_packages(exclude=["tests","scripts","docs"]),
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite='tests',
    zip_safe=False,
    include_package_data=True,
    license="MIT License",
    keywords='chemicalchecker bioactivity signatures chemoinformatics',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
