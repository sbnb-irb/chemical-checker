from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, Boolean, ForeignKey, Integer


@logged
class Datafiles(Base):
    """A Signature bla bla."""
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
        """ Method to add a new row to the table.

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
    def get(dataset):
        """ Method to query datafiles table.

        Args:
            dataset(str):The code of the dataset to get all files related to
                that dataset.
        """
        session = get_session()
        query = session.query(Datafiles).filter_by(dataset=dataset)
        res = query.all()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
