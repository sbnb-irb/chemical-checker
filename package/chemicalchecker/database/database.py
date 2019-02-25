from chemicalchecker.util import Config
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


Base = declarative_base()


def get_engine(dbname=None):
    """Method to get engine for ORM SQLALCHEMY.

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        engine
    """
    config = Config()

    if config.DB.dialect == 'sqlite':
        con = config.DB.dialect + ':///' + config.DB.file
        engine = create_engine(con, echo=True, poolclass=NullPool)
        return engine

    con = '{dialect}://{user}:{password}@{host}:{port}/{database}'.format(
        **config.DB.asdict())
    engine = create_engine(con, poolclass=NullPool)
    return engine


def get_session(dbname=None):
    """Method to get session for ORM SQLALCHEMY.

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
