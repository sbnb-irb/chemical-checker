"""Generic database functions.

These utility functions allow class from this module to get engine and
open session to the desired database. Also preppare the base class that will
be extended by each :mod:`~chemicalchecker.database` class.
"""
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from chemicalchecker.util import Config

# Construct a base class for declarative class definitions.
Base = declarative_base()


class DBConfig():
    """DB config holder."""

    def set_config(self, config=None):
        if config is None:
            config = Config()
        self.config = config
        return self


dbconfig = DBConfig().set_config()


def set_db_config(config=None):
    dbconfig.set_config(config)


def get_engine(dbname=None):
    """Get database engine.

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        engine
    """

    if dbconfig.config.DB.dialect == 'sqlite':
        con = dbconfig.config.DB.dialect + ':///' + dbconfig.config.DB.file
        engine = create_engine(con, echo=True, poolclass=NullPool)
        return engine

    params = dbconfig.config.DB.asdict()

    if dbname is not None:
        params["database"] = dbname

    con = '{dialect}://{user}:{password}@{host}:{port}/{database}'.format(
        **params)
    engine = create_engine(con, poolclass=NullPool)
    return engine


def get_session(dbname=None):
    """Get database session.

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        session
    """
    Session = sessionmaker()
    engine = get_engine(dbname)
    Session.configure(bind=engine)
    session = Session()
    return session
