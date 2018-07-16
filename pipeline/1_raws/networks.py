#!/miniconda/bin/python


# Imports

import h5py
import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import collections
import numpy as np
from sklearn.preprocessing import normalize

# Variables

dbname = ''

import checkerconfig


id_conversion    = "XXXX" # Metaphors - id_conversion.txt
file_9606        = "XXXX" # Metaphors - 9606.txt
human_proteome   = "XXXX" # Human proteome - human_proteome.tab
networks_dir = ''
table = "networks"

networks = ["string", "inbiomap", "ppidb", "pathwaycommons", "recon2"]

# Functions

def human_metaphors():

    metaphorsid_uniprot = collections.defaultdict(set)
    f = open(id_conversion, "r")
    f.next()
    dbs = set()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[1] == "SwissProt" or l[1] == "TrEMBL":
            metaphorsid_uniprot[l[2]].update([l[0]])
    f.close()

    any_human = collections.defaultdict(set)
    f = open(file_9606, "r")
    f.next()
    for l in f:
        l = l.rstrip("\n").split("\t")
        if l[3] not in metaphorsid_uniprot: continue
        if l[1] not in metaphorsid_uniprot: continue
        for po in metaphorsid_uniprot[l[3]]:
            for ph in metaphorsid_uniprot[l[1]]:
                any_human[po].update([ph])
                any_human[ph].update([ph])
    f.close()

    f = open(human_proteome, "r")
    f.next()
    for l in f:
        p = l.split("\t")[0]
        any_human[p].update([p])
    f.close()

    return any_human


def fetch_binding(any_human):

    R = Psql.qstring("SELECT inchikey, raw FROM binding", dbname)

    ACTS = collections.defaultdict(list)
    for r in R:
        for x in r[1].split(","):
            uniprot_ac, act = x.split("(")
            act = int(act.split(")")[0])
            if uniprot_ac not in any_human: continue
            hps = any_human[uniprot_ac]
            
            for hp in hps:    
                ACTS[(r[0], hp)] += [act]
    ACTS = dict((k, np.max(v)) for k,v in ACTS.iteritems())

    return ACTS


# Hotnet functions

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
    

def read_hotnet_output(ACTS):

    prots = set([k[1] for k,v in ACTS.iteritems()])

    protein_profiles = collections.defaultdict(list)

    for network in networks:
        
        # Loading network
        sm = load_matrix("%s/%s" % (networks_dir, network))
        sm.A = scale_by_non_diagonal_max(sm.A)
        myprots = prots.intersection(sm.names)
        
        # Get the protein profiles
        for prot in myprots:
            P = []
            prof = sm.get_profile(prot)
            for i in xrange(len(prof)):
                w = int(prof[i]*10) # Convert weights to integers
                if w == 0: continue
                P += [(network + "_" + sm.names[i], w)]
            protein_profiles[prot] += P

    D = collections.defaultdict(list)
    for k,v in ACTS.iteritems():
        if k[1] not in protein_profiles: continue
        for p in protein_profiles[k[1]]:
            D[k[0]] += [(p[0], p[1]*v)]

    return D


def insert_to_database(D):
    inchikey_raw = {}
    for k,v in D.iteritems():
        inchikey_raw[k] = ",".join(["%s(%d)" % (x[0], x[1]) for x in v])
    Psql.insert_raw(table, inchikey_raw,dbname)


# Main

def main():

    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,id_conversion,file_9606,human_proteome,networks_dir
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    id_conversion = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.id_conversion)
    file_9606 = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.file_9606)
    human_proteome = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.human_proteome)
    networks_dir = checkercfg.getDirectory( "networks" )

    log = logSystem(sys.stdout)
    
    log.info( "Human MetPhors")
    any_human = human_metaphors()

    log.info( "Fetch activities")
    ACTS = fetch_binding(any_human)

    log.info( "Reading Hotnet output")
    D = read_hotnet_output(ACTS)

    log.info( "Inserting to database")
    insert_to_database(D)


if __name__ == '__main__':
    main()