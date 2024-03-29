"""Molecule InChIKey-InChI mapping.

Simple table storing the correspondence between InChIKey and InChI.

Example::

    from chemicalchecker.database import Molecule
    mol = Molecule.get('RZVAJINKPMORJF-UHFFFAOYSA-N'))
    mol.inchi
    >>> 'InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)'

"""
from tqdm import trange
import sqlalchemy
from sqlalchemy import Column, Text, VARCHAR
from sqlalchemy.dialects import postgresql

from .database import Base, get_session, get_engine

from chemicalchecker.util import logged


@logged
class Molecule(Base):
    """Molecule Table class.

    Parameters:
        inchikey(str): primary key, simple unique name for the Datasource.
        inchi(str): the download link.
    """
    __tablename__ = 'molecule'
    inchikey = Column(VARCHAR(27), primary_key=True, index=True)
    inchi = Column(Text)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        Molecule.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            struct = Molecule(**kwargs)

        Molecule.__log.debug(struct.inchikey)
        session = get_session()
        session.add(struct)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunk=1000, on_conflict_do_nothing=True):
        """Add lot of rows to the table.

        This method allows to load a big amount of rows in one instruction

        Args:
            data(list): The data in list format. Each list member is a new row.
                The order is important.
            chunk(int): The size of the chunks to load data to the database.
        """
        engine = get_engine()
        with engine.begin() as conn:
            for pos in range(0, len(data), chunk):
                if on_conflict_do_nothing:
                    conn.execute(
                        postgresql.insert(Molecule.__table__).values(
                            [{"inchikey": row[0], "inchi": row[1]}
                             for row in data[pos:pos + chunk]]
                        ).on_conflict_do_nothing(
                            index_elements=[Molecule.inchikey]))
                else:
                    conn.execute(
                        Molecule.__table__.insert(),
                        [{"inchikey": row[0], "inchi": row[1]}
                            for row in data[pos:pos + chunk]]
                    )

    @staticmethod
    def get(key):
        """Method to query table."""
        session = get_session()
        query = session.query(Molecule).filter_by(inchikey=key)
        res = query.one_or_none()

        session.close()

        return res

    @staticmethod
    def get_inchikey_inchi_mapping(inchikeys, batch=10000):
        mapping = dict()
        for ink in inchikeys:
            mapping[ink] = None

        session = get_session()
        desc = 'Fetching InChIKey-InChI mapping'
        dis = len(inchikeys) < batch
        for idx in trange(0, len(inchikeys), batch, desc=desc, disable=dis):
            query = session.query(Molecule).filter(
                Molecule.inchikey.in_(inchikeys[idx:idx + batch]))
            res = query.with_entities(Molecule.inchikey, Molecule.inchi).all()
            mapping.update(dict(res))

        return mapping

    @staticmethod
    def get_missing_from_set(keys):
        size = 1000
        present = set()

        vec = list(keys)

        session = get_session()
        for pos in range(0, len(keys), size):
            query = session.query(Molecule).filter(
                Molecule.inchikey.in_(vec[pos:pos + size]))
            res = query.with_entities(Molecule.inchikey).all()
            for ele in res:
                present.add(ele[0])

        session.close()

        Molecule.__log.debug("Found already present: " + str(len(present)))

        return keys.difference(present)

    @staticmethod
    def add_missing_only(data):
        """Add data to the table if not already present.

        Args:
            data(dict): The data in dict format, containing inchikey, inchi.
        """
        list_inchikey_inchi = list()
        set_inks = set(data.keys())

        Molecule.__log.debug(
            "Size initial data to add: " + str(len(set_inks)))

        todo_iks = Molecule.get_missing_from_set(set_inks)

        Molecule.__log.debug("Size final data to add: " + str(len(todo_iks)))

        for ik, inchi in data.items():
            if ik in todo_iks:
                list_inchikey_inchi.append((ik, inchi))

        if len(list_inchikey_inchi) > 0:
            Molecule.add_bulk(list_inchikey_inchi)

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(Molecule.__tablename__)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Molecule.__table__.drop(engine)