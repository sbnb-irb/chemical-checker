#!/miniconda/bin/python

# Imports

import sys, os
import networkx as nx
import collections
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import subprocess
import h5py
import numpy as np
from sklearn.preprocessing import normalize

import checkerconfig


# Variables

chebi_molrepo  = "XXXX"
all_binary_sif = "XXXX" # all_binary.sif
pcomms         = "pcommons.tsv" # pcomms.tsv This is a temporary file that will then be used as an input for hotnet!
table = "molpathways"
dbname = ''

# Functions

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

def insert_to_database(profiles, inchikey_inchi):

    inchikey_raw = {}
    for k,v in profiles.iteritems():
        inchikey_raw[k] = ",".join(["%s(%d)" % (x[0], x[1]) for x in v])

    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])
    Psql.insert_raw(table, inchikey_raw,dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,chebi_molrepo,all_binary_sif
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    chebi_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"chebi.tsv")
    all_binary_sif = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.pathway_sif)
    
    networksdir = checkercfg.getDirectory( "networks" )
    
    outdir = os.path.join(networksdir,table)

    if  os.path.exists(outdir) == False:
        c = os.makedirs(outdir)
    
    log = logSystem(sys.stdout)
    
    log.info(  "Preparing HotNet input")

    inchikey_inchi = prepare_hotnet_input(outdir)    

    log.info(  "Running HotNet")

    run_hotnet(outdir)

    log.info(  "Reading HotNet output")

    profiles = read_hotnet_output(outdir)

    log.info(  "Insert to database")

    insert_to_database(profiles, inchikey_inchi)


if __name__ == '__main__':
    main()