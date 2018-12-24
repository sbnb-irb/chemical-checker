from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Text


def Molprop(table_name):

    DynamicBase = declarative_base(class_registry=dict())

    @logged
    class GenericMolprop(DynamicBase):
        """The Mol Properties class for the table of the same name"""
        __tablename__ = table_name
        inchikey = Column(Text, primary_key=True)
        raw = Column(Text)

        @staticmethod
        def add(kwargs):
            """ Method to add a new row to the table.

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
                    GenericMolprop.__table__.insert(),
                    [{"inchikey": row[0], "raw": row[1]}
                        for row in data[pos:pos + chunk]]
                )

        @staticmethod
        def get(key):
            """ Method to query general_properties table.


            """
            session = get_session()
            query = session.query(GenericMolprop).filter_by(inchikey=key)
            res = query.one_or_none()

            session.close()

            return res

        @staticmethod
        def _create_table():
            engine = get_engine()
            Base.metadata.create_all(engine)

    return GenericMolprop
