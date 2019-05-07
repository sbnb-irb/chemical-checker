import datetime
from time import time
import os
import math
import chemicalchecker
import numpy as np
import h5py
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util.hpc import HPC
from chemicalchecker.util.parser import DataCalculator
from .database import Base, get_session, get_engine
from .molecule import Molecule
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Text
from sqlalchemy.dialects import postgresql
from chemicalchecker.database import GeneralProp


def Calcdata(table_name):

    DynamicBase = declarative_base(class_registry=dict())
    config = Config()

    @logged
    class GenericCalcdata(DynamicBase):
        """The Mol Properties class for the table of the same name."""

        __tablename__ = table_name
        inchikey = Column(Text, primary_key=True)
        raw = Column(Text)
        dbname = config.DB.calcdata_dbname

        @staticmethod
        def add(kwargs):
            """Method to add a new row to the table.

            Args:
                kwargs(dict):The data in dictionary format .
            """
            GenericCalcdata.__log.debug(type(kwargs))
            if type(kwargs) is dict:
                prop = GenericCalcdata(**kwargs)

            GenericCalcdata.__log.debug(prop.inchikey)
            session = get_session(GenericCalcdata.dbname)
            session.add(prop)
            session.commit()
            session.close()

        @staticmethod
        def get(key):
            """Method to query general_properties table."""
            session = get_session(GenericCalcdata.dbname)
            query = session.query(GenericCalcdata).filter_by(inchikey=key)
            res = query.one_or_none()

            session.close()

            return res

        @staticmethod
        def _create_table():
            engine = get_engine(GenericCalcdata.dbname)
            DynamicBase.metadata.create_all(engine)

        @staticmethod
        def get_properties_from_list(keys):
            size = 1000
            props = set()

            session = get_session(GenericCalcdata.dbname)
            for pos in range(0, len(keys), size):
                query = session.query(GenericCalcdata).filter(
                    GenericCalcdata.inchikey.in_(keys[pos:pos + size]), GenericCalcdata.raw.isnot(None))
                res = query.with_entities(
                    GenericCalcdata.inchikey, GenericCalcdata.raw).all()
                props.update(res)

            session.close()

            return list(props)

        @staticmethod
        def get_missing_from_set(keys):
            size = 1000
            present = set()

            vec = list(keys)

            session = get_session(GenericCalcdata.dbname)
            for pos in range(0, len(keys), size):
                query = session.query(GenericCalcdata).filter(
                    GenericCalcdata.inchikey.in_(vec[pos:pos + size]))
                res = query.with_entities(GenericCalcdata.inchikey).all()
                for ele in res:
                    present.add(ele[0])

            session.close()

            GenericCalcdata.__log.debug(
                "Found already present: " + str(len(present)))

            return keys.difference(present)

        @staticmethod
        def from_inchikey_inchi(inchikey_inchi, missing_only=True, chunksize=1000):
            """Method to fill the property table from an inchikey to inchi map."""
            # calc_fn yield a list of dictionaries with keys as a molprop
            # entry

            if missing_only:
                set_inks = set(inchikey_inchi.keys())

                GenericCalcdata.__log.debug(
                    "Size initial data to add: " + str(len(set_inks)))

                todo_iks = GenericCalcdata.get_missing_from_set(set_inks)

                GenericCalcdata.__log.debug(
                    "Size final data to add: " + str(len(todo_iks)))

                dict_inchikey_inchi = {k: inchikey_inchi[k] for k in todo_iks}

            else:
                dict_inchikey_inchi = inchikey_inchi

            Molecule.add_missing_only(inchikey_inchi)

            parse_fn = DataCalculator.calc_fn(GenericCalcdata.__tablename__)
            # profile time
            t_start = time()
            engine = get_engine(GenericCalcdata.dbname)
            for chunk in parse_fn(dict_inchikey_inchi, chunksize):
                if len(chunk) == 0:
                    continue
                GenericCalcdata.__log.debug(
                    "Loading chunk of size: " + str(len(chunk)))

                engine.execute(postgresql.insert(GenericCalcdata.__table__).values(
                    chunk).on_conflict_do_nothing(index_elements=[GenericCalcdata.inchikey]))
            t_end = time()
            t_delta = str(datetime.timedelta(seconds=t_end - t_start))
            GenericCalcdata.__log.info(
                "Loading Mol properties Name %s took %s", GenericCalcdata.__tablename__, t_delta)

        @staticmethod
        def calcdata_hpc(job_path, inchikey_inchi, **kwargs):
            """Run HPC jobs to calculate data from inchikey_inchi data.

            job_path(str): Path (usually in scratch) where the script files are
                generated.
            inchikey_inchi(list): List of inchikey, inchi tuples
            cpu: Number of cores each job will use(default:1)
            wait: Wait for the job to finish (default:True)
            memory: Maximum memory the job can take in Gigabytes(default: 5)
            chunk: Number of elements per HPC job(default: 200)
            """
            # create job directory if not available
            if not os.path.isdir(job_path):
                os.mkdir(job_path)

            cpu = kwargs.get("cpu", 1)
            wait = kwargs.get("wait", True)
            memory = kwargs.get("memory", 5)
            chunk = kwargs.get("chunk", 200)

            # create script file
            cc_config = os.environ['CC_CONFIG']
            cc_package = os.path.join(chemicalchecker.__path__[0], '../')
            script_lines = [
                "import sys, os",
                "import pickle",
                "import h5py",
                # cc_config location
                "os.environ['CC_CONFIG'] = '%s'" % cc_config,
                "sys.path.append('%s')" % cc_package,  # allow package import
                "from chemicalchecker.database import Calcdata",
                "task_id = sys.argv[1]",  # <TASK_ID>
                "filename = sys.argv[2]",  # <FILE>
                "h5_file = sys.argv[3]",  # <H5 FILE>
                # load pickled data
                "inputs = pickle.load(open(filename, 'rb'))",
                # elements for current job
                "start_index = int(inputs[task_id])",
                "with h5py.File(h5_file, 'r') as hf:",
                "    inchikey_inchi = dict(hf['ik_inchi'][start_index:start_index+" + str(
                    chunk) + "])",
                # elements are indexes
                "mol = Calcdata('" + GenericCalcdata.__tablename__ + "')",
                # start import
                'mol.from_inchikey_inchi(inchikey_inchi,missing_only=False)',
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

            GenericCalcdata.__log.debug(
                "Size initial data to add: " + str(len(set_inks)))

            todo_iks = GenericCalcdata.get_missing_from_set(set_inks)

            GenericCalcdata.__log.debug(
                "Size final data to add: " + str(len(todo_iks)))

            if len(todo_iks) == 0:
                return None

            for ele in inchikey_inchi:
                if ele[0] in todo_iks:
                    list_inchikey_inchi.append((str(ele[0]), str(ele[1])))

            h5_file_name = os.path.join(job_path, "ik_inchi.h5")
            del inchikey_inchi

            with h5py.File(h5_file_name, "w") as hf:
                hf.create_dataset(
                    "ik_inchi", data=np.array(list_inchikey_inchi))

            indices = range(0, len(list_inchikey_inchi), chunk)

            params = {}

            params["num_jobs"] = len(indices)
            params["jobdir"] = job_path
            params["job_name"] = "CC_MLP_" + GenericCalcdata.__tablename__
            params["elements"] = indices
            params["wait"] = wait
            params["cpu"] = cpu
            params["memory"] = memory
            # job command
            singularity_image = Config().PATH.SINGULARITY_IMAGE
            command = "singularity exec {} python {} <TASK_ID> <FILE> {}".format(
                singularity_image, script_name, h5_file_name)
            # submit jobs
            cluster = HPC(Config())
            cluster.submitMultiJob(command, **params)

    return GenericCalcdata
