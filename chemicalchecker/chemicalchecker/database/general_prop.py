from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Integer, Text, Float


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
                GeneralProp.__table__.insert(),
                [{"inchikey": row[0], "mw": row[1], "heavy": row[2], "hetero": row[3],
                  "rings": row[4], "ringaliph": row[5], "ringarom": row[6], "alogp": row[7],
                  "mr": row[8], "hba": row[9], "hbd": row[10], "psa": row[11], "rotb": row[12],
                  "alerts_qed": row[13], "alerts_chembl": row[14], "ro5": row[15], "ro3": row[16], "qed": row[17]}
                    for row in data[pos:pos + chunk]]
            )

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
