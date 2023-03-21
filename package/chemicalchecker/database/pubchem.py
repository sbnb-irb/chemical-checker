"""PubChem synonims table.

Table for compound synonims from PubChem.
"""
from sqlalchemy import Column, Integer, Text

from .database import Base, get_session, get_engine

from chemicalchecker.util import logged


@logged
class Pubchem(Base):
    """Pubchem Table class."""
    __tablename__ = 'pubchem'
    cid = Column(Integer, primary_key=True)
    inchikey_pubchem = Column(Text)
    inchikey = Column(Text)
    name = Column(Text)
    synonyms = Column(Text)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        Pubchem.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            pubchem = Pubchem(**kwargs)

        Pubchem.__log.debug(pubchem.inchikey)
        session = get_session()
        session.add(pubchem)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunk=1000):
        """Add lot of rows to the table.

        This method allows to load a big amound of rows in one instruction

        Args:
            data(list): The data in list format. Each list member is a new row.
                The order is important.
            chunk(int): The size of the chunks to load data to the database.
        """
        engine = get_engine()
        with engine.begin() as conn:
            for pos in range(0, len(data), chunk):
                conn.execute(
                    Pubchem.__table__.insert(),
                    [{"cid": row[0], "inchikey_pubchem": row[1],
                      "inchikey": row[2], "name": row[3],
                      "synonyms": row[4]}
                        for row in data[pos:pos + chunk]]
                )

    @staticmethod
    def get(cid=None, inchikey_pubchem=None, inchikey=None, name=None,
            synonyms=None):
        """Method to query table.

        Args:
            cid(int): The cid that want to find
            inchikey_pubchem(str): The inchikey_pubchem to query
            inchikey(str): The inchikey to query
        """
        params = {}
        if cid is not None:
            params["cid"] = cid
        if inchikey_pubchem is not None:
            params["inchikey_pubchem"] = inchikey_pubchem
        if inchikey is not None:
            params["inchikey"] = inchikey
        if name is not None:
            params["name"] = name
        if synonyms is not None:
            params["synonyms"] = synonyms

        if len(params) == 0:
            return None

        session = get_session()
        query = session.query(Pubchem).filter_by(**params)
        res = query.all()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
