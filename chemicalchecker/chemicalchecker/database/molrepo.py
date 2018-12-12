"""Molrepo implementation.

The Molrepo is a repository of mappings between various textual identifier
conventions (SMILES, InChI and InChIKey) to the external source identifier.
"""
import os
import datetime
from time import time
from .database import Base, get_engine, get_session
from sqlalchemy import Column, Text, Integer
from sqlalchemy.orm import class_mapper, ColumnProperty

import chemicalchecker
from chemicalchecker.util import Parser
from chemicalchecker.util import logged
from chemicalchecker.database import Datasource
from chemicalchecker.util import Config
from chemicalchecker.util import HPC


@logged
class Molrepo(Base):
    """The Molrepo table.

    This table offer a mapping between inchikeys and different external
    compound ids (e.g. chembl, bindigdb, etc.).

    Fields:
        id(int): primary key, auto-incrementing integer.
        molrepo_name(str): the molrepo name.
        src_id(str): the download id as in the source file.
        smiles(str): simplified molecular-input line-entry system (SMILES).
        inchikey(bool): hashed version of the full InChI (SHA-256 algorithm).
        inchi(bool): International Chemical Identifier (InChI).
    """

    __tablename__ = 'molrepo'
    id = Column(Integer, primary_key=True)
    molrepo_name = Column(Text)
    src_id = Column(Text)
    smiles = Column(Text)
    inchikey = Column(Text)
    inchi = Column(Text)

    def __repr__(self):
        """String representation."""
        return self.inchikey

    @staticmethod
    def _create_table():
        engine = get_engine()
        Molrepo.metadata.create_all(engine)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Molrepo.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return engine.dialect.has_table(engine, Molrepo.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(Molrepo).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def get(inchikey):
        """Get Molrepo entries associated to the given inchikey.

        Args:
            inchikey(str): The inchikey to search for.
        """
        session = get_session()
        query = session.query(Molrepo).filter_by(inchikey=inchikey)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def get_by_molrepo_name(molrepo_name):
        """Get Molrepo entries associated to the given inchikey.

        Args:
            molrepo_name(str): The molrepo_name to search for.
        """
        session = get_session()
        query = session.query(Molrepo).filter_by(molrepo_name=molrepo_name)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def count(molrepo_name=None):
        """Get Molrepo entries associated to the given source name.

        Args:
            molrepo_name(str): The source name from `Datasource.molrepo_name`
        """
        session = get_session()
        if molrepo_name:
            query = session.query(Molrepo).filter_by(
                molrepo_name=molrepo_name).count()
        else:
            query = session.query(Molrepo).count()
        return int(query)

    @staticmethod
    def test_all_available():
        """Check if Molrepo has entries for each Datasource."""
        molrepos_ds = Datasource.get_molrepos()
        session = get_session()
        query = session.query(Molrepo).distinct(Molrepo.molrepo_name).count()
        Molrepo.__log.debug("%s Molrepos available", int(query))
        return int(query) == len(molrepos_ds)

    @staticmethod
    def from_datasource(ds):
        """Fill Molrepo table from Datasource.

        Args:
            ds(Datasource): a Datasource entry.
        """
        if not ds.available_molrepo:
            raise Exception("Datasource molrepo file not available.")
        molrepo_name = ds.molrepo_name
        Molrepo.__log.debug("Importing Datasource %s", ds)
        # parser_fn yield a list of dictionaries with keys as a molrepo entry
        parse_fn = Parser.parse_fn(ds.molrepo_parser)
        # profile time
        t_start = time()
        engine = get_engine()
        for chunk in parse_fn(ds.molrepo_path, molrepo_name, 1000):
            engine.execute(Molrepo.__table__.insert(), chunk)
        t_end = time()
        t_delta = str(datetime.timedelta(seconds=t_end - t_start))
        Molrepo.__log.info("Importing Datasource %s took %s", ds, t_delta)

    @staticmethod
    def molrepo_hpc(job_path):
        """Run HPC jobs importing all molrepos.

        job_path(str): Path (usually in scratch) where the script files are
            generated.
        """
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config = os.environ['CC_CONFIG']
        cc_package = os.path.join(chemicalchecker.__path__[0], '../')
        script_lines = [
            "import sys, os",
            "import pickle",
            "os.environ['CC_CONFIG'] = '%s'" % cc_config,  # cc_config location
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.database import Datasource",
            "from chemicalchecker.database import Molrepo",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for d in data:",  # elements are indexes
            "    ds = Datasource.get_molrepos()[d]",  # query the db
            "    Molrepo.from_datasource(ds)",  # start import
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'molrepo_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasources = Datasource.get_molrepos()
        params = {}
        params["num_jobs"] = len(all_datasources)
        params["jobdir"] = job_path
        params["job_name"] = "CC_MOLREPO"
        params["elements"] = range(len(all_datasources))
        params["wait"] = True
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
