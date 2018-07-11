'''
Created on Oct 15, 2013
    
    Connect to Postgresql

@author: mduran
'''
import psycopg2
from tqdm import tqdm
import uuid
import subprocess
import shelve
import numpy as np
import sys, os

# FUNCTIONS

def connect(d, h='aloy-dbsrv',u='sbnb-adm',p='db_adm'):
    datab = psycopg2.connect(host=h, user=u, password=p, database=d)
    return datab
    
def qstring(query, dbname):
    con = connect(dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    con.close()
    return rows

def query(query, dbname):
    con = connect(dbname)
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    con.close()
    
def escape(s):
    return s.replace("'","''")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def toStr(x):
    if x is None:
        return "NULL"
    else:
        return "'%s'" % x

def not_yet_in_table(inchikeys, table,db, chunk = 2000):
    for c in tqdm(chunker(inchikeys, chunk)):
        s = ",".join(["('%s')" % x for x in c])
        cmd = "SELECT t.inchikey FROM (VALUES %s) AS t (inchikey) LEFT JOIN %s w ON w.inchikey = t.inchikey WHERE w.inchikey IS NULL" % (s, table)
        for r in qstring(cmd, db):
            yield r[0]

# Dedicated insert functions

def insert_structures(inchikey_inchi,db, chunk = 1000):
    todos = [ik for ik in not_yet_in_table(list(inchikey_inchi.keys()), "structure",db)]
    for c in tqdm(chunker(todos, chunk)):
        s = ", ".join(["('%s', '%s')" % (k, inchikey_inchi[k]) for k in c]) 
        query("INSERT INTO structure (inchikey, inchi) VALUES %s ON CONFLICT DO NOTHING" % s, db)

    return todos

def insert_raw(table, inchikey_raw, db, chunk = 10000, truncate = False):
    if truncate:
        Psql.query("TRUNCATE %s CASCADE" % table, db)
    todos = [ik for ik in not_yet_in_table(list(inchikey_raw.keys()), table,db)]
    for c in tqdm(chunker(todos, chunk)): 
        s = ",".join(["('%s', '%s')" % (k, inchikey_raw[k]) for k in c])
        query("INSERT INTO %s (inchikey, raw) VALUES %s ON CONFLICT DO NOTHING" % (table, s), db)
