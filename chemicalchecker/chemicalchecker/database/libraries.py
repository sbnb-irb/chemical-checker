from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Integer, Text


@logged
class Libraries(Base):
    """The Libraries class for the table of the same name"""
    __tablename__ = 'library_description'
    lib = Column(Text, primary_key=True)
    files = Column(Text)
    name = Column(Text)
    description = Column(Text)
    urls = Column(Text)
    rank = Column(Integer)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        Libraries.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            lib = Libraries(**kwargs)

        Libraries.__log.debug(lib.lib)
        session = get_session()
        session.add(lib)
        session.commit()
        session.close()

    @staticmethod
    def get(lib):
        """ Method to query libraries table.

        Args:
            lib(str):The lib id to query
        """
        session = get_session()
        query = session.query(Libraries).filter_by(lib=lib)
        res = query.one_or_none()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
