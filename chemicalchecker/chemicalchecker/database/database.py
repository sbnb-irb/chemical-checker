from chemicalchecker.util import Config
from chemicalchecker.util import logged
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


Base = declarative_base()


def get_connection(dbname=None):
    """ Method to connect to PSQL DB through psycopg2

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        connection
    """
    import psycopg2
    config = Config()

    conn_dict = config.DB.asdict()
    conn_dict.pop('dialect')
    datab = psycopg2.connect(**conn_dict)
    return datab


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
    engine = create_engine(con, echo=True, poolclass=NullPool)
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


def qstring(query, dbname):
    """Method to query a PSQL DB.

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        rows: the data queried in row format
    """
    con = get_connection(dbname=dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    con.close()
    return rows


def query(query, dbname):
    """Method to query a PSQL database which returns data.

    Args:
        dbname(str):The name of DB to connect.
            If none, the name from config file is used.
    """
    con = get_connection(dbname=dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    con.close()
