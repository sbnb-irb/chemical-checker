import os
import munch
import binascii
from six import string_types
from psycopg2.extras import execute_values

from chemicalchecker.util import psql


def get_random_hash(length):
    """Returns a random hash (only hexadecimal digits) of a certain length"""
    rand_hash = binascii.hexlify(os.urandom(int(length / 2) + 1))[:length]
    return rand_hash.decode('ascii')


class UniprotKbError(Exception):
    """Error during the access to the UniprotKB database"""

    def __init__(self, msg):
        super(UniprotKbError, self).__init__(msg)


class UniprotKB(object):
    """This class provides an interface to querying the internal database UniprotKB"""

    DEFAULT_HOST = 'aloy-dbsrv.irbbarcelona.pcb.ub.es'
    DEFAULT_USER = 'uniprot_user'
    DEFAULT_PWD = 'uniprot'

    SRC_DB_GENEID = 'geneid'
    SRC_DB_ENSEMBL = 'ensembl'
    SRC_DB_GI = 'gi'
    SRC_DB_REFSEQ = 'refseq'
    SRC_DB_EMBL = 'embl'
    SRC_DB_FLYBASE = 'flybase'
    SRC_DB_HGNC = 'hgnc'
    SRC_DB_WORMBASE = 'wormbase'
    SRC_DB_SGD = 'sgd'

    UNIPROTKB_TABLES = {
        'uniprotkb_protein': [
            'uniprot_ac', 'uniprot_id', 'fullname', 'taxid', 'organism',
            'existence', 'flag', 'source', 'complete', 'reference',
            'genename', 'length', 'function'
        ]
    }

    def __init__(self, version, host=None, user=None, pwd=None, port=None):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.port = port
        self.dbname = 'uniprotkb_' + version

        self._db_conn = None

    def __del__(self):
        self.close_conn()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close_conn()

    def _get_new_connection(self):
        return psql.get_connection(self.dbname)

    @property
    def _db(self):
        if not self._db_conn:
            self._db_conn = self._get_new_connection()
        return self._db_conn

    def close_conn(self):
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None

    def get_reference_proteome(self, tax_id, only_reviewed=False):
        """Returns the set of Uniprot ACs belonging to the reference proteome
        for the organism corresponding to the tax_id."""

        query_txt = "SELECT DISTINCT uniprot_ac FROM uniprotkb_protein WHERE taxid = '%s' AND reference = 'Reference proteome'" % str(
            tax_id)
        if only_reviewed:
            query_txt += " AND source = 'sprot'"
        return psql.qstring(query_txt, self.dbname)

    def get_protein(self, uniprot_ac, limit_to_fields=None):
        """Returns the specified protein record"""

        fields = [f for f in self.UNIPROTKB_TABLES['uniprotkb_protein']
                  if not limit_to_fields or f in limit_to_fields]

        if not fields:
            raise ValueError('The list of fields is not valid, must be a subset of [%s]' % ', '.join(
                self.UNIPROTKB_TABLES['uniprotkb_protein']))

        query_txt = 'SELECT ' + \
            ', '.join(fields) + ' FROM uniprotkb_protein WHERE uniprot_ac = %s'

        cur = self._db.cursor()
        cur.execute(query_txt, (uniprot_ac,))

        results = cur.fetchall()
        if results:
            return munch.munchify({f: results[0][i] for i, f in enumerate(fields)})
        else:
            raise UniprotKbError('Uniprot AC %s not found' % uniprot_ac)

    def get_proteins(self, uniprot_acs, limit_to_fields=None):
        """Returns the records for all specified proteins."""

        list_of_acs = sorted(list(set(uniprot_acs)))

        fields = [f for f in self.UNIPROTKB_TABLES['uniprotkb_protein']
                  if not limit_to_fields or f in limit_to_fields]

        if not fields:
            raise ValueError('The list of fields is not valid, must be a subset of [%s]' % ', '.join(
                self.UNIPROTKB_TABLES['uniprotkb_protein']))

        with self._get_new_connection() as new_db_conn:
            cur = new_db_conn.cursor()
            temp_table_name = "uniprot_acs_to_link_" + get_random_hash(10)
            cur.execute('CREATE TEMP TABLE ' + temp_table_name +
                        ' ( uats TEXT PRIMARY KEY)')
            execute_values(cur, 'INSERT INTO ' + temp_table_name +
                           ' VALUES %s', [(ii,) for ii in list_of_acs])

            query_txt = 'SELECT ii.uats, ' + ', '.join(['up.%s' % f for f in fields]) + ' FROM ' + \
                temp_table_name + ' AS ii JOIN uniprotkb_protein AS up ON up.uniprot_ac = ii.uats'

            cur.execute(query_txt)

            proteins = cur.fetchall()

        return munch.munchify({p[0]: {f: p[i] for i, f in enumerate(fields, start=1)} for p in proteins})

    def pick_reference(self, uniprot_acs, organism_tax_ids=None, only_one=True):
        """
        Among a set of ambiguously mapped uniprot ACs that, supposedly, refer to the same entity, this function picks
        the "best", defined as the one corresponding to one of the organism tax ids that is:

          * the longest among the ones that are reviewed and assigned to the Complete and Reference proteome
          * if not, the longest among the ones that are reviewed and assigned to the Reference proteome
          * if not, the longest among the ones that are reviewed and assigned to the Complete proteome
          * if not, the longest among the ones that are assigned to the Complete and Reference proteome
          * if not, the longest among the ones that are assigned to the Reference proteome
          * if not, the longest among the ones that are assigned to the Complete proteome
          * if not, the longest among the ones that are reviewed
          * if not, the longest

        If *only_one* is set to false, instead of returning the "longest among..." it returns the entire list of
        proteins satisfying each condition.
        """

        def _get_longest(pps):
            if pps:
                if only_one:
                    if len(pps) == 1:
                        return next(iter(pps.keys()))
                    elif len(pps) > 1:
                        return max(pps.keys(), key=(lambda x: pps[x].length))
                else:
                    return sorted(pps.keys())
            else:
                return None

        def _adapt_output(o):
            if not only_one and isinstance(o, string_types):
                return [o]
            else:
                return o

        if not uniprot_acs:
            raise ValueError('uniprot_acs cannot be empty')

        if len(uniprot_acs) == 1:
            return _adapt_output(next(iter(uniprot_acs)))

        proteins = self.get_proteins(uniprot_acs, limit_to_fields=[
                                     'source', 'taxid', 'reference', 'complete', 'length'])

        if organism_tax_ids:
            proteins = {p: v for p, v in proteins.items() if (
                v.taxid in organism_tax_ids)}

        if not proteins:
            return None
        elif len(proteins) == 1:
            return _adapt_output(next(iter(proteins.keys())))

        sel_proteins = {p: v for p, v in proteins.items() if (v.source == 'sprot' and
                                                              v.reference == 'Reference proteome' and
                                                              v.complete == 'Complete proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p, v in proteins.items() if (v.source == 'sprot' and
                                                              v.reference == 'Reference proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p, v in proteins.items() if (v.source == 'sprot' and
                                                              v.complete == 'Complete proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p, v in proteins.items() if (v.reference == 'Reference proteome' and
                                                              v.complete == 'Complete proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p, v in proteins.items() if (
            v.reference == 'Reference proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p, v in proteins.items() if (
            v.complete == 'Complete proteome')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        sel_proteins = {p: v for p,
                        v in proteins.items() if (v.source == 'sprot')}

        p = _get_longest(sel_proteins)
        if p:
            return _adapt_output(p)

        p = _get_longest(proteins)
        if p:
            return _adapt_output(p)

    def map_secondary_to_primary(self, uniprot_ac):
        """Maps a secondary Uniprot AC to a primary one."""

        query_txt = "SELECT DISTINCT primary_ac FROM uniprotkb_secondary WHERE secondary_ac = '%s'" % str(
            uniprot_ac)
        return psql.qstring(query_txt, self.dbname)

    def map_xrefs_to_uniprot_acs(self, ids, filter_dbs=None):
        """Maps external IDs to Uniprot AC."""

        list_of_ids = sorted(list(set(ids)))

        with self._get_new_connection() as new_db_conn:
            cur = new_db_conn.cursor()
            temp_table_name = "xref_ids_to_map_" + get_random_hash(10)
            cur.execute('CREATE TEMP TABLE ' + temp_table_name +
                        ' ( id TEXT PRIMARY KEY)')
            execute_values(cur, 'INSERT INTO ' + temp_table_name +
                           ' VALUES %s', [(ii,) for ii in list_of_ids])

            if filter_dbs:
                cur.execute('SELECT uniprot_ac, db, xref FROM uniprotkb_xref AS ux JOIN ' +
                            temp_table_name + ' AS ii ON ux.xref = ii.id WHERE db IN %s', (tuple(set(filter_dbs)),))
            else:
                cur.execute('SELECT uniprot_ac, db, xref FROM ' + temp_table_name +
                            ' AS ii JOIN uniprotkb_xref AS ux ON ux.xref = ii.id')

            mapping = {}
            for uniprot_ac, db, xref in cur.fetchall():
                mapping.setdefault(xref, set()).add((uniprot_ac, db))

            return mapping

    def map_protein_to_uniref100_representative(self, uniprot_ac):
        """Returns the Uniprot ACs representative of the Uniref100 cluster to which *uniprot_ac* belongs."""

        query_txt = "SELECT uniref100_uniprot_ac FROM uniprotkb_uniref100_canonical WHERE uniprot_ac = %s" % uniprot_ac

        results = psql.qstring(query_txt, self.dbname)

        return results[0] if results else uniprot_ac

    def map_names_to_uniprot_acs(self, names, filter_sources=None, filter_taxids=None):
        """Maps names to Uniprot AC (like ORF names, for ex.)."""

        list_of_names = sorted(list(set(names)))

        with self._get_new_connection() as new_db_conn:
            cur = new_db_conn.cursor()
            temp_table_name = "names_to_map_" + get_random_hash(10)
            cur.execute('CREATE TEMP TABLE ' + temp_table_name +
                        ' ( name TEXT PRIMARY KEY)')
            execute_values(cur, 'INSERT INTO ' + temp_table_name +
                           ' VALUES %s', [(ii,) for ii in list_of_names])

            query_txt = 'SELECT ii.name, uniprot_ac, source, taxid FROM ' + \
                temp_table_name + ' AS ii JOIN uniprotkb_names AS ux ON ii.name = ux.name'
            if filter_sources or filter_taxids:
                query_txt += ' WHERE '
                if filter_sources:
                    query_txt += "source IN ('%s')" % "', '".join(filter_sources)
                    if filter_taxids:
                        query_txt += ' AND '
                if filter_taxids:
                    query_txt += "taxid IN ('%s')" % "', '".join(filter_taxids)

            cur.execute(query_txt)

            mapping = {}
            for name, uniprot_ac, source, taxid in cur.fetchall():
                mapping.setdefault(name, set()).add(
                    (uniprot_ac, source, taxid))

            return mapping
