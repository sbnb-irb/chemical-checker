from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Integer, Text, Boolean


@logged
class Dataset(Base):
    """A Signature bla bla."""
    __tablename__ = 'dataset'
    code = Column(Text, primary_key=True)
    level = Column(Text)
    coordinate = Column(Text)
    name = Column(Text)
    technical_name = Column(Text)
    description = Column(Text)
    unknowns = Column(Boolean)
    data_type = Column(Text)
    predicted = Column(Boolean)
    connectivity = Column(Boolean)
    keys = Column(Text)
    num_keys = Column(Integer)
    features = Column(Text)
    num_features = Column(Integer)
    exemplary = Column(Boolean)
    version = Column(Text)
    public = Column(Boolean)

    @staticmethod
    def add(kwargs):
        """ Method to add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format .
        """
        Dataset.__log.debug(type(kwargs))
        if type(kwargs) is dict:
            dataset = Dataset(**kwargs)

        Dataset.__log.debug(dataset.code)
        session = get_session()
        session.add(dataset)
        session.commit()
        session.close()

    @staticmethod
    def get(code):
        """ Method to query general_properties table.

        Args:
            code(str):The code of the dataset to get.
        """
        session = get_session()
        query = session.query(Dataset).filter_by(code=code)
        res = query.one_or_none()

        session.close()

        return res

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine)
