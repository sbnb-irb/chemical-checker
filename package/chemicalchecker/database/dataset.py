"""Dataset definition.

In the CC nomenclature, a dataset is determined by:


* One coordinate.

* One (typically) or multiple (eventually) sources having the same type of
(mergeable) data.

* A processing procedure yielding signatures type 0.

"""
from chemicalchecker.util import logged
from .database import Base, get_session, get_engine
from sqlalchemy import Column, Text, Boolean, ForeignKey, VARCHAR
from sqlalchemy.orm import class_mapper, ColumnProperty, relationship


@logged
class Dataset(Base):
    """The Dataset table.

    Parameters:
        dataset_code(str): primary key, Identifier of the dataset.
        level(str): The CC level.
        coordinate(str): Coordinates in the CC organization.
        name(str): Display, short-name of the dataset.
        technical_name(str): A more technical name for the dataset, suitable for chemo-/bio-informaticians.
        description(str): This field contains a long description of the dataset.
        unknowns(bool): Does the dataset contain known/unknown data.
        discrete(str): The type of data that ultimately expresses de dataset, after the pre-processing.
        keys(str): In the core CC database, most of the times this field will correspond to CPD, as the CC is centred on small molecules.
        features(str): Twe express with this field the type of biological entities.
        exemplary(bool): Is the dataset exemplary of the coordinate.
        public(bool): Is dataset public.
    """

    __tablename__ = 'dataset'
    dataset_code = Column(VARCHAR(6), primary_key=True)
    level = Column(VARCHAR(1))
    coordinate = Column(VARCHAR(2))
    name = Column(Text)
    technical_name = Column(Text)
    description = Column(Text)
    unknowns = Column(Boolean)
    discrete = Column(Boolean)
    keys = Column(VARCHAR(3))
    features = Column(VARCHAR(3))
    exemplary = Column(Boolean)
    public = Column(Boolean)
    essential = Column(Boolean)

    datasources = relationship("Datasource",
                               secondary="dataset_has_datasource",
                               lazy='joined')

    def __repr__(self):
        """String representation."""
        return self.dataset_code

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[Dataset.__table__])

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
        df = pd.read_csv(filename, delimiter=";")
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
            query = session.query(Dataset).filter_by(dataset_code=code)
            res = query.one_or_none()
        else:
            query = session.query(Dataset).distinct(Dataset.dataset_code)
            res = query.all()
        session.close()
        return res


@logged
class DatasetHasDatasource(Base):
    """Dataset-Datasource have Many-to-Many relationship."""

    __tablename__ = 'dataset_has_datasource'
    dataset_code = Column(VARCHAR(6),
                          ForeignKey("dataset.dataset_code"), primary_key=True)
    datasource_name = Column(Text,
                             ForeignKey("datasource.datasource_name"), primary_key=True)

    def __repr__(self):
        """String representation."""
        return self.dataset_code + " maps to " + self.datasource_name

    @staticmethod
    def _create_table():
        engine = get_engine()
        Base.metadata.create_all(engine, tables=[DatasetHasDatasource.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        DatasetHasDatasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return engine.dialect.has_table(engine,
                                        DatasetHasDatasource.__tablename__)

    @staticmethod
    def _table_attributes():
        attrs = [a for a in class_mapper(
            DatasetHasDatasource).iterate_properties]
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
            entry = DatasetHasDatasource(**kwargs)
        DatasetHasDatasource.__log.debug(entry)
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
        needed_cols = DatasetHasDatasource._table_attributes()
        if needed_cols != list(df.columns):
            raise Exception("Input missing columns: %s", ' '.join(needed_cols))
        # add them
        for row_nr, row in df.iterrows():
            try:
                DatasetHasDatasource.add(row.dropna().to_dict())
            except Exception as err:
                DatasetHasDatasource.__log.error(
                    "Error in line %s: %s", row_nr, str(err))
