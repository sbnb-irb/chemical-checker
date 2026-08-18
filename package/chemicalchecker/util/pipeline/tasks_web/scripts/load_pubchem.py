"""Load synonyms.

To avoid querying all the synonyms we reuse what we already stored in a
previous version of the DB and we only query missing compounds.
"""
import h5py
import sys
import time
import socket
import pubchempy
import requests
import json
import pickle
from chemicalchecker.util import psql
from chemicalchecker.util import logged
from chemicalchecker.util.parser.request_helpers import _throttle

import os
import logging
logging.log(logging.DEBUG, 'CWD: {{}}'.format(os.getcwd()))

# `pubchempy` calls `urlopen` without a timeout, so a stalled connection hangs
# forever. This job runs for days, so give every socket a ceiling.
socket.setdefaulttimeout(120)

# How many CIDs to ask synonyms for in a single request.
SYNONYM_BATCH = 200

CLASSYFIRE_URL = 'http://classyfire.wishartlab.com/entities/%s.json'

# A compound ClassyFire is known to classify, used to tell "this entity is
# broken" apart from "the service is down". Aspirin: `Acylsalicylic acids`.
CLASSYFIRE_CONTROL = 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'

# (checked_at, is_up) for the health probe below, cached for CF_HEALTH_TTL.
CF_HEALTH_TTL = 60
_CF_HEALTH = [0.0, True]

# ClassyFire allows ~12 requests/minute and answers "429" above that.
CLASSYFIRE_DELAY = 5.0
_cf_last = [0.0]


def _classyfire_throttle():
    """Block until CLASSYFIRE_DELAY has elapsed since the last ClassyFire hit."""
    wait = CLASSYFIRE_DELAY - (time.time() - _cf_last[0])
    if wait > 0:
        time.sleep(wait)
    _cf_last[0] = time.time()

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


def pubchem_call(fn, *args, **kwargs):
    """Run a `pubchempy` call paced by the shared throttle.

    `pubchempy` does its own HTTP and exposes neither the response headers
    nor any rate limiting, so we can only pace it from the outside: every
    call waits its turn on the same throttle used by `request_helpers`.
    Failures back off exponentially instead of being retried immediately --
    an immediate retry against a 503 is what gets an IP blocked.
    """
    attempts = 8
    delay = 5
    max_delay = 300
    for attempt in range(attempts):
        _throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts - 1:
                raise Exception(
                    "Too many errors when querying pubchem API: %s" % e)
            print("PubChem request failed (%s). Backing off %ds..."
                  % (e, delay))
            time.sleep(delay)

            delay = min(delay * 2, max_delay)


def _classyfire_up(timeout=30):
    """Is ClassyFire up, or is this a site-wide failure?

    A 5xx on one entity means that entity is broken; a 5xx on everything
    means the service is down and nothing should be persisted. Asking for a
    key known to classify tells the two apart. The answer is cached for
    `CF_HEALTH_TTL` seconds so a real outage does not double the request
    rate, and an unreachable control reads as "down" -- failing towards
    deferring, which is the recoverable direction.

    Only a 5xx or a failed connection counts as down. Anything else -- 429 in
    particular -- is the service answering, so it is evidence of the opposite:
    reading 429 as "down" is what deferred 238 of 271 keys on the first run
    with this probe.
    """
    now = time.time()
    if now - _CF_HEALTH[0] < CF_HEALTH_TTL:
        return _CF_HEALTH[1]
    _classyfire_throttle()
    try:
        r = requests.get(CLASSYFIRE_URL % CLASSYFIRE_CONTROL, timeout=timeout)
        up = not (500 <= r.status_code < 600)
    except Exception as e:
        print('ClassyFire health probe failed: %s' % str(e))
        up = False
    _CF_HEALTH[0], _CF_HEALTH[1] = now, up
    return up


def query_direct(ik, timeout=30, retries=2, backoff=5):
    """Return the ClassyFire `direct_parent` name for `ik`.

    Returns '' when ClassyFire answered but has no classification for `ik`
    (a definitive negative), and None when it could not be reached at all.
    Callers must keep those apart: a definitive negative is safe to persist,
    whereas persisting a failed lookup marks the compound as resolved and it
    is never queried again.

    Paced by the same throttle as the PubChem calls. A 404 means the compound
    simply isn't in ClassyFire, which is the normal outcome for a large share
    of the universe, so it returns straight away rather than burning retries.
    """
    url = CLASSYFIRE_URL % ik
    server_error = False
    for attempt in range(retries):
        _classyfire_throttle()
        try:
            server_error = False
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return ''
            server_error = 500 <= r.status_code < 600
            r.raise_for_status()
            djson = json.loads(r.text)
            parent = djson.get('direct_parent') or {}
            return parent.get('name') or ''
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff)
            else:
                print('Exception in %s: %s' % (url, str(e)))

    if server_error and _classyfire_up():
        return ''
    return None


def query_synonyms(cids):
    """Return a {cid: [synonym, ...]} map, asked for in batches.

    `Compound.synonyms` costs one request *per compound* (see pubchempy's own
    docstring), which at PubChem's latency is ~3-4 s each and made this script
    the slowest step of the web update by an order of magnitude. The
    `compound/cid/<list>/synonyms` endpoint answers for a whole batch in a
    single request, so ask it that way instead.
    """
    synonyms = dict()
    for i in range(0, len(cids), SYNONYM_BATCH):
        batch = cids[i:i + SYNONYM_BATCH]
        records = pubchem_call(pubchempy.get_synonyms, batch, 'cid')
        for rec in records:
            synonyms[rec['CID']] = rec.get('Synonym') or []
    return synonyms


def query_missing_data(missing_keys):
    """Query synonyms"""
    print('Querying missing synonyms from Pubchem.')
    input_data = pubchem_call(
        pubchempy.get_compounds, sorted(missing_keys), 'inchikey')

    # `cid`, `inchikey` and `iupac_name` are all parsed from the record that
    # `get_compounds` already returned, so reading them costs nothing. Only
    # the synonyms need going back to PubChem, and those go out in batches.
    synonyms = query_synonyms([dt.cid for dt in input_data if dt.cid])

    rows = list()
    items = set(missing_keys)
    for dt in input_data:
        ik = dt.inchikey
        if ik not in items:
            continue

        iupac_name = dt.iupac_name or ''
        syns = synonyms.get(dt.cid) or []

        name = ''
        pubchem_name = ''
        if len(syns) > 0:
            name = syns[0]
            pubchem_name = name
        if name == '' and iupac_name != '':
            name = iupac_name

        # ClassyFire caps out at ~12 requests/minute, so one call per molecule
        # is what sets this task's runtime (~5 s/mol, ~21 days for a full
        # universe pass). `direct_parent` is read nowhere in the web app -- its
        # only reachable effect is supplying a name when PubChem has none -- so
        # only spend a request when there is no name to be had otherwise.
        direct_parent = ''
        if name == '':
            direct_parent = query_direct(ik) or ''
            name = direct_parent
        new_data = (dt.cid, ik, ik, name, ';'.join(syns), pubchem_name,
                    iupac_name, direct_parent)
        rows.append(new_data)
        items.remove(ik)

    print('Found via Pubchem:', len(rows))
    print('Still without synonyms:', len(items))

    deferred = 0
    if len(items) > 0:
        print('Querying direct parent information.')
        for ik in items:
            direct_parent = query_direct(ik)
            if direct_parent is None:
                # PubChem had no record and ClassyFire could not be reached,
                # so there is nothing to store but an empty placeholder --
                # and a placeholder counts as `done` on the next run, which
                # would strand this key with no data forever. Leave it out
                # so the next pass retries it.
                deferred += 1
                continue
            name = direct_parent
            new_data = (-1, '', ik, name, '', '', '', direct_parent)
            rows.append(new_data)

    if deferred:
        print('Deferred to a later run (ClassyFire unreachable):', deferred)

    if len(rows) + deferred < len(missing_keys):
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
    failed_chunks = 0
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
        rows = []

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
                try:
                    rows += query_missing_data(missing)
                except Exception as e:
                    # A chunk that cannot be resolved must not take the
                    # remaining chunks down with it: the keys it did not
                    # insert are simply still missing from the table, so the
                    # next run of the task picks them up again.
                    print('Skipping chunk, will retry on a later run: %s' % e)
                    failed_chunks += 1
                    continue
            # insert queried and old in new db
            print(len(keys), len(rows))
            if not rows:
                continue
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

    if failed_chunks:
        print('Chunks left for a later run: %d/%d'
              % (failed_chunks, len(slices)))


run()

