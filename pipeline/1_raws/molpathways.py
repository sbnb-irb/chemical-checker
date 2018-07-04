'''

Small molecule pathways.

Essentially, metabolites.

'''

# Imports

import sys, os
import networkx as nx
import collections
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql
import subprocess
import h5py
import numpy as np
from sklearn.preprocessing import normalize

# Variables

chebi_molrepo  = "XXXX"
all_binary_sif = "XXXX" # all_binary.sif
pcomms         = "XXXX" # pcomms.tsv This is a temporary file that will then be used as an input for hotnet!
table = "molpathways"

# Functions

def prepare_hotnet_input():

    # Read ChEBI molrepo

    chebi_inchikey = collections.defaultdict(set)
    f = open(chebi_molrepo, "r")
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        chebi_inchikey[l[0]] = l[2]
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

    with open(pcomms, "w") as f:
        for e in G.edges():
            f.write("%s\t%s\n" % (e[0], e[1]))

## THIS IS PROVISIONAL!

def run_hotnet():
    cmd = "python /aloy/home/mduran/myscripts/hotnet/prepare_network.py --interactions data/pcommons.tsv --output_folder data"
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
    for j in tqdm(xrange(A.shape[1])):
        S[j,j] = np.max(S[:,j])
    S = normalize(S, norm = "max", axis = 0)
    return S

def read_hotnet_output()

    sm = load_matrix("data")
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

def insert_to_database(profiles):

    inchikey_raw = {}
    for k,v in profiles.iteritems():
        inchikey_raw[k] = ",".join(["%s(%d)" % (x[0], x[1]) for x in v])

    Psql.insert_raw(table, inchikey_raw)


# Main

def main():
    
    print "Preparing HotNet input"

    prepare_hotnet_input()    

    print "Running HotNet"

    run_hotnet()

    print "Reading HotNet output"

    profiles = read_hotnet_output()

    print "Insert to database"

    insert_to_database(profiles)


if __name__ == '__main__':
    main()