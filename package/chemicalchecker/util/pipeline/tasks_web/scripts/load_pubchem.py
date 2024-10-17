"""Load synonyms.

To avoid querying all the synonyms we reuse what we already stored in a
previous version of the DB and we only query missing compounds.
"""
import h5py
import sys
import pubchempy
import requests
import json
import pickle
from chemicalchecker.util import psql
from chemicalchecker.util import logged

import os
import logging
logging.log(logging.DEBUG, 'CWD: {{}}'.format(os.getcwd()))

INSERT = """
INSERT INTO pubchem (cid, inchikey_pubchem, inchikey, name,
synonyms, pubchem_name, iupac_name, direct_parent) VALUES %s
""".replace('\n', ' ').strip()

SELECT = """
SELECT cid, inchikey_pubchem, inchikey, name, synonyms,
pubchem_name, iupac_name, direct_parent
FROM pubchem WHERE inchikey IN (%s)
""".replace('\n', ' ').strip()


def formatting(text):
    new_text = list()
    for t in text:
        if t is None:
            new_text.append("''")
            continue
        if type(t) == int:
            if t == -1:
                new_text.append('NULL')
            else:
                new_text.append(str(t))
        else:
            if t is None:
                t = ''
            new_text.append("'" + t.replace("'", "''") + "'")
    
    row = "(" + ','.join(new_text) + ")"
    row = row.replace("('','',", "(null,'',")
    return row


def query_direct(ik):
    direct_parent = ''
    try:
        url = 'http://classyfire.wishartlab.com/entities/' + ik + '.json'
        r = requests.get(url)
        if r.status_code == 200:
            djson = json.loads(r.text)
            if 'direct_parent' in djson:
                if 'name' in djson['direct_parent']:
                    direct_parent = djson['direct_parent']['name']
                    if direct_parent is None:
                        direct_parent = ''
    except Exception as e:
        print('Exception in %s: %s' % (url, str(e)))
    return direct_parent


def query_missing_data(missing_keys):
    """Query synonyms"""
    print('Querying missing synonyms from Pubchem.')
    attempts = 10
    while attempts > 0:
        try:
            input_data = pubchempy.get_compounds(missing_keys, 'inchikey')
            break
        except Exception as e:
            print("Connection failed to REST Pubchem API. Retrying...")
            attempts -= 1

    if attempts == 0:
        raise Exception("Too many errors when querying pubchem API")

    rows = list()
    items = set(missing_keys)
    for dt in input_data:
        attempts = 10
        while attempts > 0:
            try:
                data = dt.to_dict(
                    properties=['synonyms', 'cid', 'iupac_name', 'inchikey'])
                break
            except Exception as e:
                print("Connection failed to REST Pubchem API. Retrying...")
                attempts -= 1
        if attempts == 0:
            raise Exception("Too many errors when querying pubchem API")

        ik = data["inchikey"]
        if ik not in items:
            continue

        name = ''
        pubchem_name = ''
        if len(data['synonyms']) > 0:
            name = data['synonyms'][0]
            pubchem_name = name
        if name == '' and data['iupac_name'] != '':
            name = data['iupac_name']
        direct_parent = query_direct(ik)
        if name == '' and direct_parent != '':
            name = direct_parent
        new_data = (data['cid'], ik, ik, name, ';'.join(
            data['synonyms']), pubchem_name,
            data['iupac_name'], direct_parent)
        rows.append(new_data)
        items.remove(ik)

    print('Found via Pubchem:', len(rows))
    print('Still without synonyms:', len(items))

    if len(items) > 0:
        print('Querying direct parent information.')
        for ik in items:
            name = ''
            direct_parent = query_direct(ik)
            if name == '' and direct_parent != '':
                name = direct_parent
            new_data = (-1, '', ik, name, '', '', '', direct_parent)
            #print(new_data)
            rows.append(new_data)

    if len(rows) < len(missing_keys):
        raise Exception("Not all universe is added to Pubchem table (%d/%d) " %
                        (len(rows), len(missing_keys)))
    return rows

@logged
def run():
    task_id = sys.argv[1]
    filename = sys.argv[2]
    universe = sys.argv[3]
    OLD_DB = sys.argv[4]
    DB = sys.argv[5]
    inputs = pickle.load(open(filename, 'rb'))
    slices = inputs[task_id]

    found_keys = set()
    for chunk in slices:
        """
        # read chunk of inchikeys
        with h5py.File(universe, "r") as h5:
            keys = list(h5["keys"][chunk])
        temp = [ k.decode('utf8') for k in keys ]
        # query old db
        SELECT_CHECK = "SELECT DISTINCT (inchikey) FROM pubchem WHERE inchikey IN (%s)" % ', '.join("'%s'" % k for k in temp )
        rows = psql.qstring( SELECT_CHECK, DB)
        done = set( [el[0] for el in rows] )
        found_keys.update( list(done) )
        keys = list( set(temp) - done )
        
        print( 'input:', len(temp), ' - found:', len(found_keys), ' - missing:', len(keys) )
        """
        keys = chunk
        
        if( len(keys) > 0 ):
            # Old db found keys are imported in bactch in the main task script, there is no need to re search them in old db to insert
            
            """
            query = SELECT % ', '.join("'%s'" % k for k in keys)
            rows = psql.qstring(query, OLD_DB)
            for row in rows:
                # check if what was in the db is valid!
                if row[0] is not None:
                    found_keys.add(row[2])
            """
            
            # query what's missing
            missing = set(keys).difference(found_keys)
            
            run._log.debug( f'keys in chunk: { len(keys) }' )
            run._log.debug( f'found keys: { len(found_keys) }' )
            run._log.debug( f'missing: { len(missing) }' )
            
            print('keys in chunk:', len(keys))
            print('found_keys:', len(found_keys))
            print('missing:', len(missing))
            #print(missing)
            if len(missing) > 0:
                rows += query_missing_data(missing)
            # insert queried and old in new db
            print(len(keys), len(rows))
            values = ', '.join(map(formatting, rows))
            #values = values.replace("('',","(null,")
            try:
                psql.query(INSERT % values, DB)
            except Exception as e:
                print(str(e))
                pass
                #print(str(e))
                #for row in rows:
                #    print('DEBUG:', row)
                #print(str(e))
run()

