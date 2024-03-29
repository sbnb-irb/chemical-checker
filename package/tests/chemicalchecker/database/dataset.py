"""Bioactivity Dataset definition.

This is how we define a dataset for a bioactivity space:

+-----------------------+-----------------------+-----------------------+
| Column                | Values                | Description           |
+=======================+=======================+=======================+
| Dataset_code          | e.g. ``A1.001``       | Identifier of the     |
|                       |                       | dataset.              |
+-----------------------+-----------------------+-----------------------+
| Level                 | e.g. ``A``            | The CC level.         |
+-----------------------+-----------------------+-----------------------+
| Coordinate            | e.g. ``A1``           | Coordinates in the CC |
|                       |                       | organization.         |
+-----------------------+-----------------------+-----------------------+
| Name                  | 2D fingerprints       | Display, short-name   |
|                       |                       | of the dataset.       |
+-----------------------+-----------------------+-----------------------+
| Technical name        | 1024-bit Morgan       | A more technical name |
|                       | fingerprints          | for the dataset,      |
|                       |                       | suitable for          |
|                       |                       | chemo                 |
|                       |                       | -/bio-informaticians. |
+-----------------------+-----------------------+-----------------------+
| Description           | 2D fingerprints are…  | This field contains a |
|                       |                       | long description of   |
|                       |                       | the dataset. It is    |
|                       |                       | important that the    |
|                       |                       | curator outlines here |
|                       |                       | the importance of the |
|                       |                       | dataset, why did      |
|                       |                       | he/she make the       |
|                       |                       | decision to include   |
|                       |                       | it, and what are the  |
|                       |                       | scenarios where this  |
|                       |                       | dataset may be        |
|                       |                       | useful.               |
+-----------------------+-----------------------+-----------------------+
| Unknowns              | ``True``/``False``    | Does the dataset      |
|                       |                       | contain known/unknown |
|                       |                       | data? Binding data    |
|                       |                       | from chemogenomics    |
|                       |                       | datasets, for         |
|                       |                       | example, are          |
|                       |                       | positive-unlabeled,   |
|                       |                       | so they do contain    |
|                       |                       | unknowns. Conversely, |
|                       |                       | chemical fingerprints |
|                       |                       | or gene expression    |
|                       |                       | data do not contain   |
|                       |                       | unknowns.             |
+-----------------------+-----------------------+-----------------------+
| Discrete              | ``True``/``False``    | The type of data that |
|                       |                       | ultimately expresses  |
|                       |                       | de dataset, after the |
|                       |                       | pre-processing.       |
|                       |                       | Categorical variables |
|                       |                       | are not allowed; they |
|                       |                       | must be converted to  |
|                       |                       | one-hot encoding or   |
|                       |                       | binarized. Mixed      |
|                       |                       | variables are not     |
|                       |                       | allowed, either.      |
+-----------------------+-----------------------+-----------------------+
| Keys                  | e.g. ``CPD`` (we use  | In the core CC        |
|                       | @afernandez           | database, most of the |
|                       | ``Bioteque``          | times this field will |
|                       | nomenclature). Can be | correspond to         |
|                       | ``NULL``.             | ``CPD``, as the CC is |
|                       |                       | centred on small      |
|                       |                       | molecules. It only    |
|                       |                       | makes sense to have   |
|                       |                       | keys of different     |
|                       |                       | types when we do      |
|                       |                       | connectivity          |
|                       |                       | attempts, that is,    |
|                       |                       | for example, when     |
|                       |                       | mapping disease gene  |
|                       |                       | expression            |
|                       |                       | signatures.           |
+-----------------------+-----------------------+-----------------------+
| Features              | e.g. ``GEN`` (we use  | When features         |
|                       | ``Bioteque``          | correspond to         |
|                       | nomenclature). Can be | explicit knowledge,   |
|                       | ``NULL``.             | such as proteins,     |
|                       |                       | gene ontology         |
|                       |                       | processes, or         |
|                       |                       | indications, we       |
|                       |                       | express with this     |
|                       |                       | field the type of     |
|                       |                       | biological entities.  |
|                       |                       | It is not allowed to  |
|                       |                       | mix different feature |
|                       |                       | types. Features can,  |
|                       |                       | however, have no      |
|                       |                       | type, typically when  |
|                       |                       | they come from a      |
|                       |                       | heavily-processed     |
|                       |                       | dataset, such as      |
|                       |                       | gene-expression data. |
|                       |                       | Even if we use        |
|                       |                       | ``Bioteque``          |
|                       |                       | nomenclature to the   |
|                       |                       | define the type of    |
|                       |                       | biological data, it   |
|                       |                       | is not mandatory that |
|                       |                       | the vocabularies are  |
|                       |                       | the ones used by the  |
|                       |                       | ``Bioteque``; for     |
|                       |                       | example, I can use    |
|                       |                       | non-human UniProt     |
|                       |                       | ACs, if I deem it     |
|                       |                       | necessary.            |
+-----------------------+-----------------------+-----------------------+
| Exemplary             | ``True``/``False``    | Is the dataset        |
|                       |                       | exemplary of the      |
|                       |                       | coordinate. Only one  |
|                       |                       | exemplary dataset is  |
|                       |                       | valid for each        |
|                       |                       | coordinate. Exemplary |
|                       |                       | datasets should have  |
|                       |                       | good coverage (both   |
|                       |                       | in keys space and     |
|                       |                       | feature space) and    |
|                       |                       | acceptable quality of |
|                       |                       | the data.             |
+-----------------------+-----------------------+-----------------------+
| Public                | ``True``/``False``    | Some datasets are     |
|                       |                       | public, and some are  |
|                       |                       | not, especially those |
|                       |                       | that come from        |
|                       |                       | collaborations with   |
|                       |                       | the pharma industry.  |
+-----------------------+-----------------------+-----------------------+
| Essential             | ``True``/``False``    | Essentail Datasets    |
|                       |                       | are required for      |
|                       |                       | the signaturization   |
|                       |                       | pipeline to work.     |
+-----------------------+-----------------------+-----------------------+
| Derived               | ``True``/``False``    | Dataset can be        |
|                       |                       | derived from existing |
|                       |                       | data (i.e. they come  |
|                       |                       | from an external      |
|                       |                       | datasource) or they   |
|                       |                       | are calculated and    |
|                       |                       | are virtually         |
|                       |                       | available for any     |
|                       |                       | compound (e.g. "A1"). |
+-----------------------+-----------------------+-----------------------+
| Datasources           | Foreign key to        | Data sources that are |
|                       | ``DataSource`` table. | used for generating   |
|                       |                       | signature 0 oof the   |
|                       |                       | dataset.              |
+-----------------------+-----------------------+-----------------------+

Dataset-Datasource have Many-to-Many relationshipi .i.e. a dataset can refer
to multiple datasources and one datasource can be used by many datasets.
For example ``drugbank`` is a class :mod:`~chemicalchecker.database.datasource`
that is used by both ``B1.001`` and ``E1.001`` but each of them also have
additional and different datasources.
"""
from sqlalchemy import Column, Text, Boolean, ForeignKey, VARCHAR
from sqlalchemy.orm import class_mapper, ColumnProperty, relationship
import sqlalchemy
from .database import Base, get_session, get_engine

from chemicalchecker.util import logged


@logged
class Dataset(Base):  # NS Base is a base class from SQLAlchemy, no __init__??
    """Dataset Table class.

    Parameters:
        dataset_code(str): primary key, Identifier of the dataset.
        level(str): The CC level.
        coordinate(str): Coordinates in the CC organization.
        name(str): Display, short-name of the dataset.
        technical_name(str): A more technical name for the dataset, suitable
            for chemo-/bio-informaticians.
        description(str): This field contains a long description of the
            dataset.
        unknowns(bool): Does the dataset contain known/unknown data.
        discrete(str): The type of data that ultimately expresses de dataset,
            after the pre-processing.
        keys(str): In the core CC database, most of the times this field will
            correspond to CPD, as the CC is centred on small molecules.
        features(str): Twe express with this field the type of biological
            entities.
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
    # derived = Column(Boolean)  # implemented as property

    datasources = relationship("Datasource",
                            secondary="dataset_has_datasource",
                            back_populates="datasets",
                            lazy='joined')

    def __repr__(self):
        """String representation."""
        return self.dataset_code

    def __lt__(self, other):
        return self.dataset_code < other.dataset_code

    @property
    def code(self):
        return self.dataset_code

    @property
    def derived(self):
        return len(self.datasources) > 0

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
        return sqlalchemy.inspect(engine).has_table(Dataset.__tablename__)

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

        # The boolean columns must be changed to boolean values otherwise
        # SQLalchmy passes strings
        df.unknowns = df.unknowns.apply(lambda x: False if x == 'f' else True)
        df.discrete = df.discrete.apply(lambda x: False if x == 'f' else True)
        df.exemplary = df.exemplary.apply(lambda x: False if x == 'f' else True)
        df.public = df.public.apply(lambda x: False if x == 'f' else True)
        df.essential = df.essential.apply(lambda x: False if x == 'f' else True)

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
    def get(code=None, **kwargs):
        """Get Dataset with given code.

        Args:
            code(str):The Dataset code, e.g "A1.001"
        """
        session = get_session()
        if code is not None:
            query = session.query(Dataset).filter_by(dataset_code=code,
                                                     **kwargs)
            res = query.one_or_none()
            session.close()
            return res
        else:
            query = session.query(Dataset).distinct(
                Dataset.dataset_code).filter_by(**kwargs)
            res = query.all()
            session.close()
            return sorted(res)

    @staticmethod
    def get_coordinates():
        """Get Dataset list of possible coordinates."""
        session = get_session()

        query = session.query(Dataset).distinct(Dataset.coordinate)
        res = query.all()
        session.close()
        return res


@logged
class DatasetHasDatasource(Base):
    """Dataset-Datasource relationship.

    Many-to-Many relationship.
    """
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
        Base.metadata.create_all(
            engine, tables=[DatasetHasDatasource.__table__])

    @staticmethod
    def _drop_table():
        engine = get_engine()
        DatasetHasDatasource.__table__.drop(engine)

    @staticmethod
    def _table_exists():
        engine = get_engine()
        return sqlalchemy.inspect(engine).has_table(DatasetHasDatasource.__tablename__)

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
