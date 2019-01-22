import datetime
from time import time

from chemicalchecker.util import logged
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
        def from_inchikey_inchi(inchikey_inchi):
            """Method to fill the property table from an inchikey to inchi map."""
            # calc_fn yield a list of dictionaries with keys as a molprop
            # entry
            parse_fn = PropCalculator.calc_fn(GenericMolprop.__tablename__)
            # profile time
            t_start = time()
            engine = get_engine()
            for chunk in parse_fn(inchikey_inchi, 1000):
                GenericMolprop.__log.debug(
                    "Loading chunk of size: " + str(len(chunk)))
                if GenericMolprop.__tablename__ == "physchem":
                    GeneralProp.add_bulk(chunk)
                else:
                    engine.execute(postgresql.insert(GenericMolprop.__table__).values(
                        chunk).on_conflict_do_nothing(index_elements=[GenericMolprop.inchikey]))
            t_end = time()
            t_delta = str(datetime.timedelta(seconds=t_end - t_start))
            GenericMolprop.__log.info(
                "Loading Mol properties Name %s took %s", GenericMolprop.__tablename__, t_delta)

    return GenericMolprop
