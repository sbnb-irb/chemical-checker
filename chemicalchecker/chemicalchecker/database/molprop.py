import datetime
from time import time
import os
import chemicalchecker
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util import HPC
from chemicalchecker.util import PropCalculator
from .database import Base, get_session, get_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Text
from sqlalchemy.dialects import postgresql
from chemicalchecker.database import GeneralProp


def Molprop(table_name):

    DynamicBase = declarative_base(class_registry=dict())

    @logged
    class GenericMolprop(DynamicBase):
        """The Mol Properties class for the table of the same name."""

        __tablename__ = table_name
        inchikey = Column(Text, primary_key=True)
        raw = Column(Text)

        @staticmethod
        def add(kwargs):
            """Method to add a new row to the table.

            Args:
                kwargs(dict):The data in dictionary format .
            """
            GenericMolprop.__log.debug(type(kwargs))
            if type(kwargs) is dict:
                prop = GenericMolprop(**kwargs)

            GenericMolprop.__log.debug(prop.inchikey)
            session = get_session()
            session.add(prop)
            session.commit()
            session.close()

        @staticmethod
        def get(key):
            """Method to query general_properties table."""
            session = get_session()
            query = session.query(GenericMolprop).filter_by(inchikey=key)
            res = query.one_or_none()

            session.close()

            return res

        @staticmethod
        def _create_table():
            engine = get_engine()
            Base.metadata.create_all(engine)

        @staticmethod
        def get_missing_from_set(keys):
            size = 1000
            present = set()

            vec = list(keys)

            session = get_session()
            for pos in range(0, len(keys), size):
                query = session.query(GenericMolprop).filter(
                    GenericMolprop.inchikey.in_(vec[pos:pos + size]))
                res = query.with_entities(GenericMolprop.inchikey).all()
                for ele in res:
                    present.add(ele[0])

            session.close()

            GenericMolprop.__log.debug(
                "Found already present: " + str(len(present)))

            return keys.difference(present)

        @staticmethod
        def from_inchikey_inchi(inchikey_inchi):
            """Method to fill the property table from an inchikey to inchi map."""
            # calc_fn yield a list of dictionaries with keys as a molprop
            # entry
            parse_fn = PropCalculator.calc_fn(GenericMolprop.__tablename__)
            # profile time
            t_start = time()
            engine = get_engine()
            for chunk in parse_fn(inchikey_inchi, 1000):
                if len(chunk) == 0:
                    continue
                GenericMolprop.__log.debug(
                    "Loading chunk of size: " + str(len(chunk)))
                if GenericMolprop.__tablename__ == GeneralProp.__tablename__:
                    GeneralProp.add_bulk(chunk)
                else:
                    engine.execute(postgresql.insert(GenericMolprop.__table__).values(
                        chunk).on_conflict_do_nothing(index_elements=[GenericMolprop.inchikey]))
            t_end = time()
            t_delta = str(datetime.timedelta(seconds=t_end - t_start))
            GenericMolprop.__log.info(
                "Loading Mol properties Name %s took %s", GenericMolprop.__tablename__, t_delta)

        @staticmethod
        def molprop_hpc(job_path, inchikey_inchi, **kwargs):
            """Run HPC jobs importing all molrepos.

            job_path(str): Path (usually in scratch) where the script files are
                generated.
            inchikey_inchi(list): List of inchikey, inchi tuples
            cpu: Number of cores each job will use(default:1)
            wait: Wait for the job to finish (default:True)
            memory: Maximum memory the job can take in Gigabytes(default: 10)
            """
            # create job directory if not available
            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            cpu = kwargs.get("cpu", 1)
            wait = kwargs.get("wait", True)
            memory = kwargs.get("memory", 10)

            # create script file
            cc_config = os.environ['CC_CONFIG']
            cc_package = os.path.join(chemicalchecker.__path__[0], '../')
            script_lines = [
                "import sys, os",
                "import pickle",
                # cc_config location
                "os.environ['CC_CONFIG'] = '%s'" % cc_config,
                "sys.path.append('%s')" % cc_package,  # allow package import
                "from chemicalchecker.database import Molprop",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                # elements for current job
                "inchikey_inchi = dict(inputs[task_id])",
                # elements are indexes
                "mol = Molprop('" + GenericMolprop.__tablename__ + "')",
                'mol.from_inchikey_inchi(inchikey_inchi)',  # start import
                "print('JOB DONE')"
            ]
            script_name = os.path.join(job_path, 'molprop_script.py')
            with open(script_name, 'w') as fh:
                for line in script_lines:
                    fh.write(line + '\n')

            set_inks = set()
            list_inchikey_inchi = list()

            for ele in inchikey_inchi:
                if ele[0] is None:
                    continue
                set_inks.add(ele[0])

            GenericMolprop.__log.debug(
                "Size initial data to add: " + str(len(set_inks)))

            todo_iks = GenericMolprop.get_missing_from_set(set_inks)

            GenericMolprop.__log.debug(
                "Size final data to add: " + str(len(todo_iks)))

            if len(todo_iks) == 0:
                return None

            for ele in inchikey_inchi:
                if ele[0] in todo_iks:
                    list_inchikey_inchi.append(ele)

            params = {}
            if GenericMolprop.__tablename__ == "fp3d":
                params["num_jobs"] = len(list_inchikey_inchi) / 200
            else:
                params["num_jobs"] = len(list_inchikey_inchi) / 2000
            params["jobdir"] = job_path
            params["job_name"] = "CC_MLP_" + GenericMolprop.__tablename__
            params["elements"] = list_inchikey_inchi
            params["wait"] = wait
            params["cpu"] = cpu
            params["memory"] = memory
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
                singularity_image, script_name)
            # submit jobs
            cluster = HPC(Config())
            cluster.submitMultiJob(command, **params)
            return cluster

    return GenericMolprop
