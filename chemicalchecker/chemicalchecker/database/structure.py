from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text
from sqlalchemy.dialects import postgresql

@logged
class Structure(Base):
    """The structure class for the table of the same name"""
    __tablename__ = 'structure'
    inchikey = Column(Text, primary_key=True)
    inchi = Column(Text)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        Structure.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            struct = Structure(**kwargs)

        Structure.__log.debug(struct.inchikey)
        session = get_session()
        session.add(struct)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunk=1000, on_conflict_do_nothing=True):
        """ Method to add a lot of rows to the table.

            This method allows to load a big amount of rows in one instruction

        Args:
            data(list): The data in list format. Each list member is a new row. It is important the order.
            chunk(int): The size of the chunks to load data to the database.
        """
        engine = get_engine()
        for pos in range(0, len(data), chunk):
            if on_conflict_do_nothing:
                engine.execute(postgresql.insert(Structure.__table__).values(
                    [{"inchikey": row[0], "inchi": row[1]}
                     for row in data[pos:pos + chunk]]).on_conflict_do_nothing(index_elements=[Structure.inchikey]))
            else:
                engine.execute(
                    Structure.__table__.insert(),
                    [{"inchikey": row[0], "inchi": row[1]}
                        for row in data[pos:pos + chunk]]
                )

    @staticmethod
    def get(key):
        """ Method to query structure table.


        """
        session = get_session()
        query = session.query(Structure).filter_by(inchikey=key)
        res = query.one_or_none()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
