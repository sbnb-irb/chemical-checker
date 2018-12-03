import os
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, Boolean, ForeignKey, Integer

from chemicalchecker.util import logged
from chemicalchecker.util import Downloader
from chemicalchecker.util import Config


@logged
class Datafiles(Base):
    """The Datafile table."""
    __tablename__ = 'datafiles'
    id = Column(Integer, primary_key=True)
    url = Column(Text)
    dataset = Column(Text, ForeignKey("dataset.code"),
                     nullable=False)  # foreign key
    permanent = Column(Boolean)
    enabled = Column(Boolean)
    username = Column(Text)
    password = Column(Text)
    download_dir = Column(Text)
    download_file = Column(Text)
    description = Column(Text)

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        Datafiles.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            datafiles = Datafiles(**kwargs)

        Datafiles.__log.debug(datafiles.code)
        session = get_session()
        session.add(datafiles)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunk=1000):
        """ Method to add a lot of rows to the table.

            This method allows to load a big amound of rows in one instruction

        Args:
            data(list): The data in list format. Each list member is a new row. it is important the order.
            chunk(int): The size of the chunks to load data to the database.
        """
        engine = get_engine()
        for pos in range(0, len(data), chunk):

            engine.execute(
                Datafiles.__table__.insert(),
                [{"url": row[0], "dataset": row[1], "permanent": row[2], "enabled": row[3],
                  "username": row[4], "password": row[5], "download_dir": row[6], "download_file": row[7],
                  "description": row[8]} for row in data[pos:pos + chunk]]
            )

    @staticmethod
    def get(dataset=None):
        """ Get datafiles associated to the given dataset.

        Args:
            dataset(str):The dataset code, e.g "A1.001"
        """
        session = get_session()
        if dataset is not None:
            query = session.query(Datafiles).filter_by(dataset=dataset)
        else:
            query = session.query(Datafiles)
        res = query.all()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)

    def download(self):
        """Download the datafile."""
        # create download string
        if self.username and self.password:
            protocol, address = self.link.split('//')
            url = "%s//%s@%s:%s".format(protocol,
                                        self.username, self.password, address)
        else:
            url = self.url
        # create download path
        cfg = Config()
        data_path = os.path.join(cfg.PATH.CC_DATA, self.download_dir)
        down = Downloader(url, data_path)
        down.download()
