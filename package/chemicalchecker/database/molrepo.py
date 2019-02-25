"""Molrepo implementation.

The Molrepo is a repository of mappings between various textual identifier
conventions (SMILES, InChI and InChIKey) to the external source identifier.
"""
import os
import datetime
from time import time
from .database import Base, get_engine, get_session
from sqlalchemy import Column, Text
from sqlalchemy.orm import class_mapper, ColumnProperty
from sqlalchemy.dialects import postgresql


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
        id(str): primary key, src_id + "_" + molrepo_name.
        molrepo_name(str): the molrepo name.
        src_id(str): the download id as in the source file.
        smiles(str): simplified molecular-input line-entry system (SMILES).
        inchikey(bool): hashed version of the full InChI (SHA-256 algorithm).
        inchi(bool): International Chemical Identifier (InChI).
    """

    __tablename__ = 'molrepo'
    id = Column(Text, primary_key=True)
    molrepo_name = Column(Text, index=True)
    src_id = Column(Text)
    smiles = Column(Text)
    inchikey = Column(Text, index=True)
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
    def get_by_molrepo_name(molrepo_name, only_raw=False):
        """Get Molrepo entries associated to the given inchikey.

        Args:
            molrepo_name(str): The molrepo_name to search for.
            only_raw(bool): Only get the raw values without the whole object(default:false)
        """
        session = get_session()
        query = session.query(Molrepo).filter_by(molrepo_name=molrepo_name)
        if only_raw:
            res = query.with_entities(
                Molrepo.molrepo_name, Molrepo.src_id, Molrepo.smiles, Molrepo.inchikey, Molrepo.inchi).all()
        else:
            res = query.all()
        session.close()
        return res

    @staticmethod
    def get_fields_by_molrepo_name(molrepo_name, fields=None):
        """Get specified column fields from a molrepo_name in raw format(tuples).

        Args:
            molrepo_name(str): The molrepo_name to search for.
            fields(list): List of field names. If None, all fields.
        """

        if fields is None:
            return Molrepo.get_by_molrepo_name(molrepo_name, True)

        cols = Molrepo._table_attributes()
        query_fields = []

        for field in fields:
            if field in cols:
                query_fields.append(field)

        if len(query_fields) == 0:
            return None

        session = get_session()
        query = session.query(Molrepo).filter_by(molrepo_name=molrepo_name)
        res = query.with_entities(*[eval("Molrepo.%s" % f)
                                    for f in query_fields]).all()

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
    def from_molrepo_name(molrepo_name):
        """Fill Molrepo table from a molrepo name.

        Args:
            molrepo_name(str): a molrepo name.
        """
        datasources = Datasource.get_molrepos(molrepo_name)
        if len(datasources) == 0:
            raise Exception(
                "Molrepo name %s file not available.", molrepo_name)
        molrepo_path = []
        molrepo_parser = datasources[0].molrepo_name
        for ds in datasources:
            Molrepo.__log.debug("Importing Datasource %s", ds)
            # Download datasource
            ds.download()
            molrepo_path.append(ds.molrepo_path)

        # parser_fn yield a list of dictionaries with keys as a molrepo entry
        parse_fn = Parser.parse_fn(molrepo_parser)
        # profile time
        t_start = time()
        engine = get_engine()
        for chunk in parse_fn(molrepo_path, molrepo_name, 1000):
            if len(chunk) == 0:
                continue
            engine.execute(postgresql.insert(Molrepo.__table__).values(
                chunk).on_conflict_do_nothing(index_elements=[Molrepo.id]))
            # engine.execute(Molrepo.__table__.insert(), chunk)
        t_end = time()
        t_delta = str(datetime.timedelta(seconds=t_end - t_start))
        Molrepo.__log.info(
            "Importing Molrepo Name %s took %s", molrepo_name, t_delta)

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
        # Download datasource
        ds.download()
        # parser_fn yield a list of dictionaries with keys as a molrepo entry
        parse_fn = Parser.parse_fn(ds.molrepo_name)
        # profile time
        t_start = time()
        engine = get_engine()
        for chunk in parse_fn([ds.molrepo_path], molrepo_name, 1000):
            if len(chunk) == 0:
                continue
            engine.execute(postgresql.insert(Molrepo.__table__).values(
                chunk).on_conflict_do_nothing(index_elements=[Molrepo.id]))
            # engine.execute(Molrepo.__table__.insert(), chunk)
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
            "    Molrepo.from_molrepo_name(d)",  # start import
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'molrepo_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasources = Datasource.get_molrepos()
        molrepos_names = set()
        for ds in all_datasources:
            molrepos_names.add(ds.molrepo_name)

        params = {}
        params["num_jobs"] = len(molrepos_names)
        params["jobdir"] = job_path
        params["job_name"] = "CC_MOLREPO"
        params["elements"] = list(molrepos_names)
        params["wait"] = True
        params["check_error"] = False
        params["memory"] = 16
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC(Config())
        cluster.submitMultiJob(command, **params)
        return cluster
