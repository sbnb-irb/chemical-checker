"""Dataset definition.

In the CC nomenclature, a dataset is determined by:


* One coordinate.

* One (typically) or multiple (eventually) sources having the same type of
(mergeable) data.

* A processing procedure yielding signatures type 0.

"""
from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, Boolean, ForeignKey, Integer
from sqlalchemy.orm import class_mapper, ColumnProperty, relationship, backref


@logged
class Dataset(Base):
    """The Dataset table.

    Parameters:
        code(str): primary key, simple unique code for the Dataset.
        level(str): TODO add field description.
        coordinate(str): TODO add field description.
        name(str): TODO add field description.
        technical_name(str): TODO add field description.
        description(str): TODO add field description.
        unknowns(bool): TODO add field description.
        data_type(str): TODO add field description.
        predicted(bool): TODO add field description.
        connectivity(bool): TODO add field description.
        keys(str): TODO add field description.
        num_keys(int): TODO add field description.
        features(str): TODO add field description.
        exemplary(bool): TODO add field description.
        version(str): TODO add field description.
        public(bool): TODO add field description.
    """

    __tablename__ = 'dataset'
    code = Column(Text, primary_key=True)
    level = Column(Text)
    coordinate = Column(Text)
    name = Column(Text)
    technical_name = Column(Text)
    description = Column(Text)
    unknowns = Column(Boolean)
    is_discrete = Column(Boolean)
    predicted = Column(Boolean)
    connectivity = Column(Boolean)
    keys = Column(Text)
    num_keys = Column(Integer)
    features = Column(Text)
    num_features = Column(Integer)
    exemplary = Column(Boolean)
    version = Column(Text)
    public = Column(Boolean)

    datasources = relationship("Datasource",
                               secondary="map_dataset_datasource",
                               lazy='joined')

    def __repr__(self):
        """String representation."""
        return self.code

    @staticmethod
    def _create_table():
        engine = get_engine()
        Dataset.metadata.create_all(engine)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        Dataset.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return engine.dialect.has_table(engine, Dataset.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(Dataset).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs]
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        if type(kwargs) is dict:
            entry = Dataset(**kwargs)
        Dataset.__log.debug(entry)
        session = get_session()
        session.add(entry)
        session.commit()
        session.close()

    @staticmethod
    def from_csv(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # check columns
        needed_cols = Dataset._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                Dataset.add(row.dropna().to_dict())
            except Exception as err:
                Dataset.__log.error(
                    "Error in line %s: %s", row_nr, str(err))

    @staticmethod
    def get(code=None):
        """Get Dataset with given code.

        Args:
            code(str):The Dataset code, e.g "A1.001"
        """
        session = get_session()
        if code is not None:
            query = session.query(Dataset).filter_by(code=code)
            res = query.one_or_none()
        else:
            query = session.query(Dataset).distinct(Dataset.code)
            res = query.all()
        session.close()
        return res


@logged
class MapDatasetDatasource(Base):
    """Dataset-Datasource have Many-to-Many relationship."""

    __tablename__ = 'map_dataset_datasource'
    id = Column(Integer, primary_key=True)
    dataset_code = Column(Text,
                          ForeignKey("dataset.code"), primary_key=True)
    datasource_name = Column(Text,
                             ForeignKey("datasource.name"), primary_key=True)

    def __repr__(self):
        """String representation."""
        return self.dataset_code + " maps to " + self.datasource_name

    @staticmethod
    def _create_table():
        engine = get_engine()
        MapDatasetDatasource.metadata.create_all(engine)

    @staticmethod
    def _drop_table():
        engine = get_engine()
        MapDatasetDatasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return engine.dialect.has_table(engine,
                                        MapDatasetDatasource.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(
            MapDatasetDatasource).iterate_properties]
        col_attrs = [a.key for a in attrs if isinstance(a, ColumnProperty)]
        input_attrs = [a for a in col_attrs if a != 'id']
        return input_attrs

    @staticmethod
    def add(kwargs):
        """Add a new row to the table.

        Args:
            kwargs(dict):The data in dictionary format.
        """
        if type(kwargs) is dict:
            entry = MapDatasetDatasource(**kwargs)
        MapDatasetDatasource.__log.debug(entry)
        session = get_session()
        session.add(entry)
        session.commit()
        session.close()

    @staticmethod
    def from_csv(filename):
        """Add entries from CSV file.

        Args:
            filename(str): Path to a CSV file.
        """
        import pandas as pd
        df = pd.read_csv(filename)
        # check columns
        needed_cols = MapDatasetDatasource._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                MapDatasetDatasource.add(row.dropna().to_dict())
            except Exception as err:
                MapDatasetDatasource.__log.error(
                    "Error in line %s: %s", row_nr, str(err))
