from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, or_
from sqlalchemy.dialects import postgresql
from tqdm import tqdm


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
    def get_inchikey_inchi_mapping(inchikeys, batch=10000):
        mapping = dict()
        for ink in inchikeys:
            mapping[ink] = None

        engine = get_engine()
        table = Structure.__table__
        for idx in tqdm(range(0, len(inchikeys), batch)):
            conditions = [table.columns.inchikey ==
                          ink for ink in inchikeys[idx:idx + batch]]
            query = table.select(or_(*conditions))
            res = engine.execute(query).fetchall()
            mapping.update(dict(res))

        return mapping

    @staticmethod
    def get_missing_from_set(keys):
        size = 1000
        present = set()

        vec = list(keys)

        session = get_session()
        for pos in range(0, len(keys), size):
            query = session.query(Structure).filter(
                Structure.inchikey.in_(vec[pos:pos + size]))
            res = query.with_entities(Structure.inchikey).all()
            for ele in res:
                present.add(ele[0])

        session.close()

        Structure.__log.debug("Found already present: " + str(len(present)))

        return keys.difference(present)

    @staticmethod
    def add_missing_only(data):
        """ Method to add data to the table that is not already present.

            This method allows to load only the data that is not already present.

        Args:
            data(list): The data in list format, containing inchikey, inchi tuples.
        """
        list_inchikey_inchi = list()
        set_inks = set()

        for ele in data:
            if ele[0] is None:
                continue
            set_inks.add(ele[0])

        Structure.__log.debug(
            "Size initial data to add: " + str(len(set_inks)))

        todo_iks = Structure.get_missing_from_set(set_inks)

        Structure.__log.debug("Size final data to add: " + str(len(todo_iks)))

        for ele in data:
            if ele[0] in todo_iks:
                list_inchikey_inchi.append(ele)

        if len(list_inchikey_inchi) > 0:
            Structure.add_bulk(list_inchikey_inchi)

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
