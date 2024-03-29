"""Datasource definition.

A Datasource is a source of raw data. Typically comes in form of an url to an
external resource. This class offer a mean to standardize raw data collection
using the :mod:`~chemicalchecker.util.download` class and interfacing them
with the Dataset table.
"""
import os
import sqlalchemy
from sqlalchemy import Column, Text, Boolean
from sqlalchemy.orm import class_mapper, ColumnProperty, relationship

from .database import Base, get_session, get_engine

from chemicalchecker.util import logged, Config
from chemicalchecker.util.download import Downloader
from chemicalchecker.util.hpc import HPC


@logged
class Datasource(Base):
    """Datasource table class.

    Parameters:
        name(str): primary key, simple unique name for the Datasource.
        url(str): the download link.
        user(str): few downloads require credentials.
        password(str): few downloads require credentials.
        description(str): free text description of the resource.
        filename(str): optional, a `molrepo` name. NB this name is the
            value of `Molrepo.molrepo_name`  also is defininf the `Parser`
            that will be used.
        calcdata(bool): the datasource is actually from one of the calculated
            data.
    """

    __tablename__ = 'datasource'
    datasource_name = Column(Text, primary_key=True)
    description = Column(Text)
    is_db = Column(Boolean)
    url = Column(Text)
    username = Column(Text)
    password = Column(Text)
    filename = Column(Text)
    calcdata = Column(Boolean)

    datasets = relationship("Dataset",
                            secondary="dataset_has_datasource",
                            back_populates="datasources",
                            lazy='joined')

    molrepos = relationship("Molrepo",
                            secondary="molrepo_has_datasource",
                            back_populates="datasources",
                            lazy='joined')

    def __repr__(self):
        """String representation."""
        return self.datasource_name

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[Datasource.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Datasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(Datasource.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(Datasource).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs]
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        if type(kwargs) is dict:
            datasource = Datasource(**kwargs)
        Datasource.__log.debug(datasource)
        session = get_session()
        session.add(datasource)
        session.commit()
        session.close()

    @staticmethod
    def from_csv(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)

        # NS: the last column has to be changed to boolean values otherwise
        # SQLalchmy passes strings
        df.calcdata = df.calcdata.apply(lambda x: False if x == 'f' else True)
        df.is_db = df.is_db.apply(lambda x: False if x == 'f' else True)

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
    def get(name=None, dbname=None):
        """Get Datasources associated to the given dataset.

        Args:
            name(str):The Datasource name, e.g "chebi"
        """
        params = {}

        if name is not None:
            params["datasource_name"] = name
        
        if(dbname!=None):
            session = get_session(dbname)
        else:
            session = get_session()
            
        if len(params) == 0:
            query = session.query(Datasource).distinct(
                Datasource.datasource_name)
        else:
            query = session.query(Datasource).filter_by(**params)

        res = query.all()
        session.close()
        return res

    @staticmethod
    def test_all_valid_url():
        """Check if all Datasources urls are valid."""
        testable_ds = [ds for ds in Datasource.get() if not ds.username]
        return all([ds.valid_url for ds in testable_ds])

    @staticmethod
    def test_all_downloaded(only_essential=False):
        """Check if all Datasources have been downloaded.

        Args:
            only_essential(bool): Check only datasources that are essential
        """
        if only_essential:
            datasources = set()
            for ds in Datasource.get():
                for dset in ds.datasets:
                    if dset.essential:
                        datasources.add(ds)
                        break
                for molrepo in ds.molrepos:
                    if molrepo.essential:
                        datasources.add(ds)
                        break
            return all([ds.available for ds in datasources])
        else:
            return all([ds.available for ds in Datasource.get()])

    @property
    def data_path(self):
        """Check if Datasource is available."""
        return os.path.join(Config().PATH.CC_DATA, self.datasource_name)

    @property
    def available(self):
        """Check if Datasource is available."""
        if os.path.isdir(self.data_path):
            self.__log.info("%s AVAILABLE", self)
            return True
        else:
            self.__log.warning("%s NOT AVAILABLE", self)
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
        """
        if self.username and self.password:
            protocol, address = self.url.split('//')
            url = "{}//{}:{}@{}".format(protocol,
                                        self.username.replace('@', '%40'),
                                        self.password, address)
        else:
            url = self.url
        """
        url = self.url
        # call the downloader
        if self.is_db:
            dbname = self.datasource_name
        else:
            dbname = None
        cc_config = os.environ['CC_CONFIG']
        cfg = Config(cc_config)
        down = Downloader(url, self.data_path, dbname=dbname,
                          file=self.filename, tmp_dir=cfg.PATH.CC_TMP, username= self.username, password=self.password)
        down.download()

    @staticmethod
    def download_hpc(job_path, only_essential=False, **kwargs):
    #Error: Tuple doesn't have get atttribute
    #def download_hpc(job_path, only_essential=False, *kwargs):
        """Run HPC jobs downloading the resources.

        Args:
            job_path(str): Path (usually in scratch) where the script files are
                generated.
            only_essential(bool):Download only the essential datasources
                (default: False).
        """
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        script_lines = [
            "import sys, os",
            "import pickle",
            "from chemicalchecker.database import Datasource",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "data = inputs[task_id]",  # elements for current job
            "for d in data:",  # elements are indexes
            "    ds = Datasource.get(name=d)",  # query the db
            "    ds[0].download()",  # start download
            "print('JOB DONE')"
        ]
        script_name = os.path.join(job_path, 'download_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters
        if only_essential:
            all_datasources = set()
            for ds in Datasource.get():
                for dset in ds.datasets:
                    if dset.essential:
                        all_datasources.add(ds)
                        break
                for molrepo in ds.molrepos:
                    if molrepo.essential:
                        all_datasources.add(ds)
                        break
        else:
            all_datasources = Datasource.get()

        ds_names = []
        for ds in all_datasources:
            ds_names.append(ds.datasource_name)

        params = {}
        params["num_jobs"] = len(ds_names)
        params["jobdir"] = job_path
        params["job_name"] = "CC_DOWNLOAD"
        params["elements"] = ds_names
        params["wait"] = True
        params["cpu"] = 4
        params["mem_by_core"] = 20
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
            " singularity exec {} python {} <TASK_ID> <FILE>"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        return cluster
