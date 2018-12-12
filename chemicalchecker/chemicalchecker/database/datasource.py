"""Datasource definition.

A Datasource is a source of raw data. Typically comes in form of an url to an
external resource. This class offer a mean to standardize raw data collection.
"""
import os
from glob import glob
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, Boolean, ForeignKey, Integer
from sqlalchemy.orm import class_mapper, ColumnProperty

import chemicalchecker
from chemicalchecker.util import logged
from chemicalchecker.util import Downloader
from chemicalchecker.util import Config
from chemicalchecker.util import HPC


@logged
class Datasource(Base):
    """The Datasource table.

    Fields:
        name(str): primary key, simple unique name for the Datasource.
        url(str): the download link.
        dataset(str): foreign key to the dataset using the Datasource.
        permanent(bool): whether the download is permanent (not updated).
        enabled(bool): flag that allow us to keep historical records.
        user(str): few downloads require credentials.
        password(str): few downloads require credentials.
        description(str): free text description of the resource.
        molrepo_name(str): optional, a `molrepo` name. NB this name is the
            value of `Molrepo.molrepo_name`
        molrepo_file(str): optional, a specific file in the Datasource that
            will be used to fill the `molrepo` table. Can also be a directory.
        molrepo_parser(str): optional, a `Parser` class able to parse the
            `molrepo_file`.
    """

    __tablename__ = 'datasource'
    id = Column(Integer, primary_key=True)
    name = Column(Text)
    url = Column(Text)
    dataset = Column(Text, ForeignKey("dataset.code"), nullable=False)
    permanent = Column(Boolean)
    enabled = Column(Boolean)
    user = Column(Text)
    password = Column(Text)
    description = Column(Text)
    molrepo_name = Column(Text)
    molrepo_file = Column(Text)
    molrepo_parser = Column(Text)
    is_db = Column(Boolean)

    def __repr__(self):
        """String representation."""
        return self.name

    @staticmethod
    def _create_table():
        engine = get_engine()
        Datasource.metadata.create_all(engine)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Datasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return engine.dialect.has_table(engine, Datasource.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(Datasource).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        Datasource.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            datasource = Datasource(**kwargs)
        Datasource.__log.debug(datasource.name)
        session = get_session()
        session.add(datasource)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunks=1000):
        """Method to add a lot of rows to the table.

        This method allows to load a big amount of rows in one instruction.

        Args:
            data(list): The data in list format. Each element will result in a
                new row. The order is important.
            chunks(int): The size of the chunks to load data to the database.
        """
        engine = get_engine()
        for pos in range(0, len(data), chunks):
            engine.execute(
                Datasource.__table__.insert(),
                [{"name": row[0],
                  "url": row[1],
                  "dataset": row[2],
                  "permanent": row[3],
                  "enabled": row[4],
                  "user": row[5],
                  "password": row[6],
                  "description": row[7],
                  "molrepo_name": row[8],
                  "molrepo_file": row[9]} for row in data[pos:pos + chunks]]
            )

    @staticmethod
    def from_cvs(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # check columns
        needed_cols = Datasource._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                Datasource.add(row.dropna().to_dict())
            except Exception as err:
                Datasource.__log.error(
                    "Error in line %s: %s", row_nr, str(err))

    @staticmethod
    def get(dataset=None):
        """Get Datasources associated to the given dataset.

        Args:
            dataset(str):The dataset code, e.g "A1.001"
        """
        session = get_session()
        if dataset is not None:
            query = session.query(Datasource).filter_by(dataset=dataset)
        else:
            query = session.query(Datasource).distinct(Datasource.name)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def get_molrepos():
        """Get Datasources associated to a molrepo."""
        session = get_session()
        query = session.query(Datasource).filter(
            ~(Datasource.molrepo_parser == '')).distinct(
            Datasource.molrepo_parser)
        res = query.all()
        session.close()
        return res

    @staticmethod
    def test_all_valid_url():
        """Check if all Datasources urls are valid."""
        testable_ds = [ds for ds in Datasource.get() if not ds.user]
        return all([ds.valid_url for ds in testable_ds])

    @staticmethod
    def test_all_downloaded():
        """Check if all Datasources have been downloaded."""
        return all([ds.available for ds in Datasource.get()])

    @staticmethod
    def test_all_molrepo():
        """Check if all molrepo files are available."""
        molrepo_ds = [ds for ds in Datasource.get() if ds.molrepo_file]
        return all([ds.available_molrepo for ds in molrepo_ds])

    @property
    def data_path(self):
        """Check if Datasource is available."""
        return os.path.join(Config().PATH.CC_DATA, self.name)

    @property
    def molrepo_path(self):
        """Build path to molrepo file or directory."""
        if not self.molrepo_file:
            self.__log.warning("Datasource %s has no molrepo file.", self)
            return None
        repo_path = os.path.join(self.data_path, self.molrepo_file)
        if '*' in self.molrepo_file:
            # resolve the path
            paths = glob(repo_path)
            if len(paths) > 1:
                raise Exception("`*` in %s molrepo_file is ambiguous.", self)
            repo_path = paths[0]
        return repo_path.encode('ascii', 'ignore')

    @property
    def available_molrepo(self):
        """Check if Datasource molrepo is available."""
        self.__log.debug("Checking %s", self.molrepo_path)
        if os.path.isfile(self.molrepo_path) or \
                os.path.isdir(self.molrepo_path):
            self.__log.info("%s AVAILABLE", self)
            return True
        else:
            self.__log.warning("%s FAILED", self)
            return False

    @property
    def available(self):
        """Check if Datasource is available."""
        if os.path.isdir(self.data_path):
            self.__log.info("%s AVAILABLE", self)
            return True
        else:
            self.__log.warning("%s FAILED", self)
            return False

    @property
    def valid_url(self):
        """Check if Datasource url is valid."""
        try:
            Downloader.validate_url(self.url)
            self.__log.info("%s AVAILABLE", self)
            return True
        except Exception as err:
            self.__log.warning("%s FAILED %s", self, str(err))
            return False

    def download(self, force=False):
        """Download the Datasource.

        force(bool): Force download overwriting previous download.
        """
        # check if already downloaded
        if not force and self.available:
            self.__log.warning("Datasource available, skipping download.")
            return
        # create download string
        if self.user and self.password:
            protocol, address = self.url.split('//')
            url = "{}//{}:{}@{}".format(protocol,
                                        self.user.replace('@', '%40'),
                                        self.password, address)
        else:
            url = self.url
        # call the downloader
        if self.is_db:
            dbname = self.name
        else:
            dbname = None
        down = Downloader(url, self.data_path, dbname=dbname,
                          dbfile=self.molrepo_file)
        down.download()

    @staticmethod
    def download_hpc(job_path):
        """Run HPC jobs downloading the resources.

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
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for d in data:",  # elements are indexes
            "    ds = Datasource.get()[d]",  # query the db
            "    ds.download()",  # start download
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'download_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        all_datasources = Datasource.get()
        params = {}
        params["num_jobs"] = len(all_datasources)
        params["jobdir"] = job_path
        params["job_name"] = "CC_DOWNLOAD"
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
