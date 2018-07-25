#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw,checkJobResultsForErrors
import Psql
from gaussian_scale_impute import scaleimpute
import numpy as np
import collections
from cmapPy.pandasGEXpress import parse
import subprocess
import h5py
import time
import uuid
import math
from scipy.stats import rankdata
from multiprocessing import Pool

import checkerconfig



# Variables

mini_sig_info_file = "XXXX" # data/new/mini_sig_info.tsv

table = "transcript"


# Functions


def parse_level(downloadsdir,signaturesdir):
    
    touchstone = set()
    with open(os.path.join(downloadsdir,"GSE92742_Broad_LINCS_pert_info.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            trt = l[2]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]: continue
            if l[3] == '0': continue
            touchstone.add(l[0])
            
    # Gene symbols (same for LINCS I and II)
    
    genes = {}
    with open(os.path.join(downloadsdir,"GSE92742_Broad_LINCS_gene_info.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.split("\t")
            genes[l[0]] = l[1]

    
    sig_info_ii = {}
    with open(os.path.join(downloadsdir,"GSE70138_Broad_LINCS_sig_info*.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in touchstone:
                v = 1
            else:
                v = 0
            sig_info_ii[l[0]] = (l[1], l[3], l[4], v, 2)
    
    # Signature metrics        
    
    sigs = collections.defaultdict(list)
    with open(os.path.join(downloadsdir,"GSE70138_Broad_LINCS_sig_metrics*.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")[1:]
            if float(l[1]) < 0.2: continue
            trt = l[5]
            sig_id = l[0]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]: continue
            if sig_id not in sig_info_ii: continue
            v = sig_info_ii[sig_id]
            tas = float(l[6])
            nsamp = int(l[-1])
            phase = 2
            sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]
            
            
    sig_info_i = {}
    with open(os.path.join(downloadsdir,"GSE92742_Broad_LINCS_sig_info.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[1] in touchstone:
                v = 1
            else:
                v = 0
            sig_info_i[l[0]] = (l[1], l[3], l[4], v, 1)
    
    with open(os.path.join(downloadsdir,"GSE92742_Broad_LINCS_sig_metrics.txt"), "r") as f:
        f.next()
        for l in f:
            l = l.rstrip("\n").split("\t")
            if float(l[4]) < 0.2: continue
            trt = l[3]
            if trt not in ["trt_cp", "trt_sh.cgs", "trt_oe"]: continue
            sig_id = l[0]
            if sig_id not in sig_info_i: continue
            v = sig_info_i[sig_id]
            tas = float(l[8])
            nsamp = int(l[-1])
            phase = 1
            sigs[(v[0], v[2])] += [(sig_id, trt, tas, nsamp, phase)]


    def get_exemplar(v):
        s = [x for x in v if x[3] >= 2 and x[3] <= 6]
        if not s:
            s = v
        sel = None
        max_tas = 0.
        for x in s:
            if not sel:
                sel = (x[0], x[-1])
                max_tas = x[2]
            else:
                if x[2] > max_tas:
                    sel = (x[0], x[-1])
                    max_tas = x[2]
        return sel

    sigs = dict((k, get_exemplar(v)) for k,v in sigs.iteritems())
    
    cids = []
    with open(mini_sig_info_file, "w") as f:
        for k,v in sigs.iteritems():
            if v[1] == 1:
                x = sig_info_i[v[0]]
            else:
                x = sig_info_ii[v[0]]
            f.write("%s\t%s\t%s\t%s\t%d\t%d\n" % (v[0], x[0], x[1], x[2], x[3], v[1]))
            cids += [(v[0], v[1])]
            
            

    gtcx_i  = os.path.join(downloadsdir,"GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx")
    gtcx_ii = os.path.join(downloadsdir,"GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328*.gctx")
        
    genes_i  = [genes[r[0]] for r in parse.parse(gtcx_i , cid = [[x[0] for x in cids if x[1] == 1][0]]).data_df.iterrows()]
    genes_ii = [genes[r[0]] for r in parse.parse(gtcx_ii, cid = [[x[0] for x in cids if x[1] == 2][0]]).data_df.iterrows()] # Just to make sure.
    
    for cid in cids:
        if cid[1] == 1:
            expr = np.array(parse.parse(gtcx_i, cid = [cid[0]]).data_df).ravel()
            genes = genes_i
        elif cid[1] == 2:
            expr = np.array(parse.parse(gtcx_ii, cid = [cid[0]]).data_df).ravel()
            genes = genes_ii
        else:
            continue
        R  = zip(genes, expr)
        R  = sorted(R, key=lambda tup: -tup[1])
        with h5py.File(os.path.join(signaturesdir,"%s.h5" % cid[0]), "w") as hf:
            hf.create_dataset("expr", data = [float(r[1]) for r in R])
            hf.create_dataset("gene", data = [r[0] for r in R])



# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  

    global mini_sig_info_file
    
    signaturesdir = checkercfg.getDirectory( "signatures" )
   

    if os.path.exists(signaturesdir) == False:
        c = os.makedirs(signaturesdir)
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    mini_sig_info_file = os.path.join(signaturesdir,'mini_sig_info.tsv')
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
    
    
    
    log.info(  "Parsing")
    parse_level(downloadsdir,signaturesdir)
    
    

if __name__ == '__main__':
    main()