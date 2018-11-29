from chemicalchecker.util import Config
from chemicalchecker.util import logged
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import psycopg2


Base = declarative_base()

config = Config()


def get_connection(dbname=None):
    """ Method to connect to PSQL DB through psycopg2

    Args:
        dbname(str):The name of DB to connect. If none, the name from config file is used .
    Res:
        connection
    """
    if dbname is None:
        dbname = config.DB.main_DB
    datab = psycopg2.connect(host=config.DB.host, user=config.DB.username,
                             password=config.DB.password, database=dbname)
    return datab


def get_engine(dbname=None):
    """ Method to get engine for ORM SQLALCHEMY

    Args:
        dbname(str):The name of DB to connect. If none, the name from config file is used .
    Res:
        engine
    """
    if config.DB.dialect == 'sqlite':
        con = config.DB.dialect + ':///' + config.DB.file
    else:
        if dbname is None:
            dbname = config.DB.main_DB
        con = config.DB.dialect + '://' + config.DB.username + ':' + config.DB.password + \
            '@' + config.DB.host + ':' + config.DB.port + '/' + dbname
    # engine = create_engine('sqlite:///orm_in_detail.sqlite', echo=True)
    engine = create_engine(con, echo=True, poolclass=NullPool)
    return engine


def get_session(dbname=None):
    """ Method to get session for ORM SQLALCHEMY

    Args:
        dbname(str):The name of DB to connect. If none, the name from config file is used .
    Res:
        session
    """
    Session = sessionmaker()
    engine = get_engine(dbname)
    Session.configure(bind=engine)
    session = Session()
    return session


def qstring(query, dbname):
    """ Method to query a PSQL DB

    Args:
        dbname(str):The name of DB to connect. If none, the name from config file is used .
    Res:
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
    """ Method to query a PSQL database which returns data

    Args:
        dbname(str):The name of DB to connect. If none, the name from config file is used .
    """
    con = get_connection(dbname=dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    con.close()
