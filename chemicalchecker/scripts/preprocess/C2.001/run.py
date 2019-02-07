import os
import sys
import argparse
import networkx as nx
import collections
import h5py
import numpy as np

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.database import Molrepo

from chemicalchecker.util import logged

# Variables

def prepare_hotnet_input(outdir):

    # Read ChEBI molrepo

    chebi_inchikey = collections.defaultdict(set)
    inchikey_inchi = {}
    f = open(chebi_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        chebi_inchikey[l[0]] = l[2]
        inchikey_inchi[l[2]] = l[3]

    f.close()

    # Read graph

    G = nx.Graph()

    with open(all_binary_sif, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[0] in chebi_inchikey and l[2] in chebi_inchikey:
                ik1 = chebi_inchikey[l[0]]
                ik2 = chebi_inchikey[l[2]]
                G.add_edge(ik1, ik2)

    with open(os.path.join(outdir,pcomms), "w") as f:
        for e in G.edges():
            f.write("%s\t%s\n" % (e[0], e[1]))

    return inchikey_inchi
## THIS IS PROVISIONAL!

def run_hotnet(outdir):
    currentDir = os.path.dirname(os.path.abspath( __file__ ))
    cmd = "python " + currentDir + "/../0_downloads/prepare_network.py --interactions " + os.path.join(outdir,pcomms) + " --output_folder " + outdir + " -p " + checkerconfig.HOTNET_PATH
    print cmd
    subprocess.Popen(cmd, shell = True).wait()

## ### ###


# Hotnet-specific functions

class Sm:
    def __init__(self, A, names):
        self.A = A
        self.names = names
    def get_profile(self, n):
        if n not in self.names: return None
        return self.A[:, self.names.index(n)]
    
def load_matrix(net_folder):
    f = open(net_folder+"/idx2node.tsv", )
    names = [l.rstrip("\n").split("\t")[1] for l in f]
    f.close()
    f = h5py.File(net_folder+"/similarity_matrix.h5")
    A = f['PPR'].value
    f.close()
    return Sm(A, names)

def network_impact(sm, node_scores):
    P, S = [], []
    for n,s in node_scores.iteritems():
        prof = sm.get_profile(sm, n)
        if not prof: continue
        P += [prof]
        S += [s]
    P = np.array(P)
    
def scale_by_non_diagonal_max(A):
    S = A[:]
    np.fill_diagonal(S, 0.)
    for j in xrange(A.shape[1]):
        S[j,j] = np.max(S[:,j])
    S = normalize(S, norm = "max", axis = 0)
    return S

def read_hotnet_output(outdir):

    sm = load_matrix(outdir)
    sm.A = scale_by_non_diagonal_max(sm.A)

    profiles = collections.defaultdict(list)
    for ik in sm.names:
        P = []
        prof = sm.get_profile(ik)
        for i in xrange(len(prof)):
            w = int(prof[i]*10) # Convert weights to integers
            if w == 0: continue
            P += [(sm.names[i], w)]
        profiles[ik] += P

    return profiles

# Parse arguments


def get_parser():
    description = 'Run preprocess script.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str,
                        required=False, default='.', help='Output file')
    return parser


@logged
def main():

    args = get_parser().parse_args(sys.argv[1:])

    dataset_code = 'C2.001'  # os.path.dirname(os.path.abspath(__file__))[-6:]

    dataset = Dataset.get(dataset_code)

    map_files = {}

    for ds in dataset.datasources:
        map_files[ds.name] = ds.data_path

    main._log.debug(
        "Running preprocess for dataset " + dataset_code + ". Saving output in " + args.output_file)

    chebi_obo = os.path.join(map_files["chebi"], "chebi.obo")

    main._log.info(  "Preparing HotNet input")

    inchikey_inchi = prepare_hotnet_input(outdir)    

    log.info(  "Running HotNet")

    run_hotnet(outdir)

    log.info(  "Reading HotNet output")

    profiles = read_hotnet_output(outdir)

    main._log.info("Saving raw data")

    keys = []
    words = set()
    for k in sorted(inchikey_paths.keys()):
        keys.append(str(k))
        words.update(inchikey_paths[k])

    orderwords = list(words)
    raws = np.zeros((len(keys), len(orderwords)), dtype=np.int8)
    wordspos = {k: v for v, k in enumerate(orderwords)}

    for i, k in enumerate(keys):
        for word in inchikey_paths[k]:
            raws[i][wordspos[word]] = 1

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(keys))
        hf.create_dataset("V", data=raws)
        hf.create_dataset("features", data=np.array(orderwords))


if __name__ == '__main__':
    main()
