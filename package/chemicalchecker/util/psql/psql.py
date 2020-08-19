"""Basic connection and query fuctions for PostgreSQL."""
try:
    import psycopg2
except ImportError:
    raise ImportError("requires psycopg2 " +
                      "http://initd.org/psycopg/")

from chemicalchecker.util import Config


def get_connection(dbname=None):
    """Return the connection to a PSQL DB.

    Args:
        dbname (str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        connection
    """
    config = Config()
    conn_dict = config.DB.asdict()
    conn_dict.pop('dialect', None)
    conn_dict.pop('calcdata_dbname', None)
    conn_dict.pop('uniprot_db_version', None)
    conn_dict.update({"database": dbname})
    datab = psycopg2.connect(**conn_dict)
    return datab


def qstring(query, dbname):
    """Method to query a PSQL DB.

    Args:
        dbname (str):The name of DB to connect.
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


def qstring_cur(query, dbname):
    """Method to query a PSQL DB.

    Args:
        dbname (str):The name of DB to connect.
            If none, the name from config file is used.
    Returns:
        rows: the data queried in row format
    """
    con = get_connection(dbname=dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    return cur


def query(query, dbname):
    """Method to query a PSQL database which returns data.

    Args:
        dbname (str):The name of DB to connect.
            If none, the name from config file is used.
    """
    con = get_connection(dbname=dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    con.close()
