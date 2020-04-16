import h5py
import sys
import pubchempy
import requests
import json
import pickle
from chemicalchecker.util import psql

INSERT = "INSERT INTO pubchem (cid, inchikey_pubchem, inchikey, name, synonyms, pubchem_name, iupac_name, direct_parent) VALUES %s"

SELECT = "SELECT cid, inchikey_pubchem, inchikey, name, synonyms, pubchem_name, iupac_name, direct_parent FROM pubchem WHERE inchikey IN (%s)"


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

    return "(" + ','.join(new_text) + ")"


def query_direct(ik):

    direct_parent = ''

    try:

        r = requests.get(
            'http://classyfire.wishartlab.com/entities/' + ik + '.json')

        if r.status_code == 200:
            djson = json.loads(r.text)
            direct_parent = djson["direct_parent"]["name"]
            if direct_parent is None:
                direct_parent = ''
            # print direct_parent
    except Exception as e:
        print(str(e))

    return direct_parent


def query_missing_data(missing_keys):

    attempts = 10
    while attempts > 0:

        try:
            input_data = pubchempy.get_compounds(missing_keys, 'inchikey')
            break
        except Exception as e:
            print "Connection failed to REST Pubchem API. Retrying..."
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
                print "Connection failed to REST Pubchem API. Retrying..."
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

        new_data = (data['cid'], ik, ik, name, ';'.join(data['synonyms']), pubchem_name, data[
                    'iupac_name'], direct_parent)

        rows.append(new_data)

        items.remove(ik)

    print len(items), len(rows)

    if len(items) > 0:

        for ik in items:

            print len(rows), ik

            name = ''
            direct_parent = query_direct(ik)

            if name == '' and direct_parent != '':
                name = direct_parent

            new_data = (-1, '', ik, name, '', '', '', direct_parent)

            rows.append(new_data)

    if len(rows) < len(missing_keys):
        raise Exception("Not all universe is added to Pubchem table (%d/%d) " %
                        (len(rows), len(missing_keys)))

    return rows

task_id = sys.argv[1]
filename = sys.argv[2]
universe = sys.argv[3]
OLD_DB = sys.argv[4]
DB = sys.argv[5]
inputs = pickle.load(open(filename, 'rb'))
slices = inputs[task_id]


found_keys = set()

for chunk in slices:
    with h5py.File(universe, "r") as h5:
        keys = list(h5["keys"][chunk])

    query = SELECT % keys
    query = query.replace("[", "")
    query = query.replace("]", "")
    rows = psql.qstring(query, OLD_DB)
    for row in rows:
        found_keys.add(row[2])

    missing = set(keys).difference(found_keys)

    print len(missing), len(keys), len(found_keys)
    print missing

    if len(missing) > 0:

        rows += query_missing_data(missing)

    print len(keys), len(rows)

    values = ', '.join(map(formatting, rows))

    try:

        psql.query(INSERT % values, DB)
    except Exception, e:

        print(e)
