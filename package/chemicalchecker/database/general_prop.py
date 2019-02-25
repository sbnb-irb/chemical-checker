from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Integer, Text, Float, String
from sqlalchemy.dialects import postgresql


@logged
class GeneralProp(Base):
    """The general_properties class for the table of the same name"""
    __tablename__ = 'physchem'
    inchikey = Column(Text, primary_key=True)
    mw = Column(Integer)
    heavy = Column(Integer)
    hetero = Column(Integer)
    rings = Column(Integer)
    ringaliph = Column(Integer)
    ringarom = Column(Integer)
    alogp = Column(Float)
    mr = Column(Float)
    hba = Column(Integer)
    hbd = Column(Integer)
    psa = Column(Float)
    rotb = Column(Integer)
    alerts_qed = Column(Integer)
    alerts_chembl = Column(Integer)
    ro5 = Column(Integer)
    ro3 = Column(Integer)
    qed = Column(Float)
    raw = Column(String)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        GeneralProp.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            prop = GeneralProp(**kwargs)

        GeneralProp.__log.debug(prop.inchikey)
        session = get_session()
        session.add(prop)
        session.commit()
        session.close()

    @staticmethod
    def add_bulk(data, chunk=1000, on_conflict_do_nothing=True):
        """ Method to add a lot of rows to the table.

            This method allows to load a big amount of rows in one instruction

        Args:
            data(list): The data in list format. The elements of the list can be a dictionary or another list. If list the order is important.
            chunk(int): The size of the chunks to load data to the database.
            on_conflict_do_nothing(bool): If data already present in table do nothing
        """
        engine = get_engine()
        if type(data[0]) == dict:
            is_dict = True
        else:
            is_dict = False
        for pos in range(0, len(data), chunk):
            if on_conflict_do_nothing:
                if is_dict:
                    engine.execute(postgresql.insert(GeneralProp.__table__).values(
                        data[pos:pos + chunk]).on_conflict_do_nothing(index_elements=[GeneralProp.inchikey]))
                else:
                    engine.execute(postgresql.insert(GeneralProp.__table__).values(
                        [{"inchikey": row[0], "mw": row[1], "heavy": row[2], "hetero": row[3],
                          "rings": row[4], "ringaliph": row[5], "ringarom": row[6], "alogp": row[7],
                          "mr": row[8], "hba": row[9], "hbd": row[10], "psa": row[11], "rotb": row[12],
                          "alerts_qed": row[13], "alerts_chembl": row[14], "ro5": row[15], "ro3": row[16], "qed": row[17], "raw": row[18]}
                         for row in data[pos:pos + chunk]]).on_conflict_do_nothing(index_elements=[GeneralProp.inchikey]))
            else:
                if is_dict:
                    engine.execute(postgresql.insert(
                        GeneralProp.__table__).values(data[pos:pos + chunk]))
                else:
                    engine.execute(
                        GeneralProp.__table__.insert(),
                        [{"inchikey": row[0], "mw": row[1], "heavy": row[2], "hetero": row[3],
                          "rings": row[4], "ringaliph": row[5], "ringarom": row[6], "alogp": row[7],
                          "mr": row[8], "hba": row[9], "hbd": row[10], "psa": row[11], "rotb": row[12],
                          "alerts_qed": row[13], "alerts_chembl": row[14], "ro5": row[15], "ro3": row[16], "qed": row[17], "raw": row[18]}
                            for row in data[pos:pos + chunk]])

    @staticmethod
    def get(key):
        """ Method to query general_properties table.


        """
        session = get_session()
        query = session.query(GeneralProp).filter_by(inchikey=key)
        res = query.one_or_none()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
