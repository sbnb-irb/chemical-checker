from chemicalchecker.util import Config
from chemicalchecker.util import logged
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import psycopg2


Base = declarative_base()

config = Config()


def get_connection():
    datab = psycopg2.connect(host=config.DB.host, user=config.DB.username,
                             password=config.DB.password, database=config.DB.main_DB)
    return datab


def get_engine():
    if config.DB.dialect == 'sqlite':
        con = config.DB.dialect + ':///' + config.DB.file
    else:
        con = config.DB.dialect + '://' + config.DB.username + ':' + config.DB.password + \
            '@' + config.DB.host + ':' + config.DB.port + '/' + config.DB.main_DB
    #engine = create_engine('sqlite:///orm_in_detail.sqlite', echo=True)
    engine = create_engine(con, echo=True, poolclass=NullPool)
    return engine


def get_session():
    Session = sessionmaker()
    engine = get_engine()
    Session.configure(bind=engine)
    session = Session()
    return session


@logged
class Database(object):
    """A class to query the cc database bla bla."""
    Base = None

    def __init__(self):
        self.__log.debug('Database class')
