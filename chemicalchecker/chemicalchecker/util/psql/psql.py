from chemicalchecker.util import Config


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
    conn_dict.update({"database": dbname})
    datab = psycopg2.connect(**conn_dict)
    return datab


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


def qstring_cur(query, dbname):
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
    return cur


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
