import sys
import argparse
import networkx as nx
import collections
import h5py


from chemicalchecker.util import logged
from chemicalchecker.database import Datasource
from chemicalchecker.util import psql
from chemicalchecker.database import Molrepo

# Variables
chembl_dbname = "chembl"

# Parse arguments


def parse_chembl():

    # Read molrepo

    molrepos = Molrepo.get("chembl")
    chemblid_inchikey = {}
    inchikey_inchi = {}
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        chemblid_inchikey[molrepo.src_id] = molrepo.inchikey
        inchikey_inchi[molrepo.inchikey] = molrepo.inchi

    def is_active(r):
        if r[3] >= 5:
            return True
        if r[2] == "Active" or r[2] == "active":
            return True
        return False

    # Query

    cur = psql.qstring('''
    SELECT md.chembl_id, cseq.accession, act.activity_comment, act.pchembl_value

    FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t

    WHERE

    (ass.src_id != 1 OR (ass.src_id = 1 AND ass.assay_type = 'F')) AND
    md.molregno = act.molregno AND
    act.assay_id = ass.assay_id AND
    ass.tid = t.tid AND
    t.component_id = cseq.component_id AND
    cseq.accession IS NOT NULL
    ''', chembl_dbname)

    R = []
    for r in cur:
        if r[0] not in chemblid_inchikey:
            continue
        if not is_active(r):
            continue
        R += [(chemblid_inchikey[r[0]], r[1])]

    # Use ChEMBL hierarchy

    S = psql.qstring(
        "SELECT protein_class_id, parent_id, pref_name FROM protein_classification", chembl_dbname)

    G = nx.DiGraph()

    for s in S:
        G.add_edge(s[1], s[0])  # The tree

    S = psql.qstring("SELECT cs.accession, cc.protein_class_id FROM component_sequences cs, component_class cc WHERE cs.component_id = cc.component_id AND cs.accession IS NOT NULL", chembl_dbname)

    class_prot = collections.defaultdict(list)

    for s in S:
        class_prot[s[0]] += [s[1]]

    classes = set([x for k, v in class_prot.iteritems() for x in v])
    class_paths = collections.defaultdict(set)
    for c in classes:
        path = set()
        p = nx.all_simple_paths(G, 0, c)
        for sp in p:
            path.update(sp)
        class_paths[c] = path

    T = set()
    for r in R:
        T.update([(r[0], r[1], inchikey_inchi[r[0]])])
        if r[1] not in class_prot:
            continue
        path = set()
        for c in class_prot[r[1]]:
            path.update(class_paths[c])
        for p in path:
            T.update([(r[0], "Class:%d" % p, inchikey_inchi[r[0]])])

    return T


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset = 'B5.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    files = Datasource.get(dataset)

    map_files = {}

    for f in files:
        map_files[f] = f.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset + ". Saving output in " + args.output_file)

    main._log.info("Parsing ChEMBL")

    T = parse_chembl()
    main._log.info("Saving raw data")
    inchikey_raw = collections.defaultdict(list)
    for t in T:
        inchikey_raw[t[0]] += [t[1]]

    keys = []
    raws = []
    for k in sorted(inchikey_raw.iterkeys()):
        raws.append(",".join([x for x in inchikey_raw[k]]))
        keys.append(k)

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=keys)
        hf.create_dataset("V", data=raws)


if __name__ == '__main__':
    main()
