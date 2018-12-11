"""Molrepo implementation.

The Molrepo is a repository of mappings between various compound molecular
conentions (smile, inchi and inchikeys) to the external source identifier.
"""
import datetime
from time import time
from .database import Base, get_engine, get_session
from sqlalchemy import Column, Text, Integer

from chemicalchecker.util import Parser
from chemicalchecker.util import logged


@logged
class Molrepo(Base):
    """The Molrepo table.

    This table offer a mapping between inchikeys and different external
    compound ids (e.g. chembl, bindigdb, etc.).

    Fields:
        id(int): primary key, auto-incrementing integer.
        molrepo_name(str): the molrepo name.
        src_id(str): the download id as in the source file.
        smile(str): smile formula.
        inchikey(bool): inchikey hash.
        inchi(bool): inchi.
    """

    __tablename__ = 'molrepo'
    id = Column(Integer, primary_key=True)
    molrepo_name = Column(Text)
    src_id = Column(Text)
    smile = Column(Text)
    inchikey = Column(Text)
    inchi = Column(Text)

    @staticmethod
    def _create_table():
        engine = get_engine()
        Molrepo.metadata.create_all(engine)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Molrepo.__table__.drop(engine)

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
