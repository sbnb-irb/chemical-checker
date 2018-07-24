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
import gaussianize as g
from scipy.stats import rankdata
from multiprocessing import Pool

import checkerconfig



# Variables

lincs_molrepo = "XXXX" # LINCS molrepo file
mini_sig_info_file = "XXXX" # data/new/mini_sig_info.tsv
ik_matrices = "XXXX" # data/new/ik_matrices/
consensus   = "consensus.h5" # consensus.h5 - Let's talk about where to save it!
dbname = ''

table = "transcript"

SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')

# Functions


def parse_level(downloaddir,signaturesdir):
    
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
            genes[int(l[0])] = l[1]

    
    sig_info_ii = {}
    with open(os.path.join(downloadsdir,"GSE70138_Broad_LINCS_sig_info*"), "r") as f:
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
        
    genes_i  = [genes[r[0]] for r in parse(gtcx_i , cid = [[x[0] for x in cids if x[1] == 1][0]]).data_df.iterrows()]
    genes_ii = [genes[r[0]] for r in parse(gtcx_ii, cid = [[x[0] for x in cids if x[1] == 2][0]]).data_df.iterrows()] # Just to make sure.
    
    for cid in cids:
        if cid[1] == 1:
            expr = np.array(parse(gtcx_i, cid = [cid[0]]).data_df).ravel()
            genes = genes_i
        elif cid[1] == 2:
            expr = np.array(parse(gtcx_ii, cid = [cid[0]]).data_df).ravel()
            genes = genes_ii
        else:
            continue
        R  = zip(genes, expr)
        R  = sorted(R, key=lambda tup: -tup[1])
        with h5py.File(os.path.join(signaturesdir,"%s.h5" % cid[0]), "w") as hf:
            hf.create_dataset("expr", data = [float(r[1]) for r in R])
            hf.create_dataset("gene", data = [r[0] for r in R])

def read_l1000(connectivitydir):
    
    with open(lincs_molrepo, "r") as f:
        pertid_inchikey = {}
        inchikey_inchi = {}
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            pertid_inchikey[l[0]] = l[2]
            inchikey_inchi[l[2]] = l[3]   

    # Read signature data

    touchstones = set()
    siginfo = {}
    with open(mini_sig_info_file, "r") as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if int(l[4]) == 1:
                touchstones.update([l[1]])
            siginfo[l[0]] = l[1]
            
    inchikey_sigid = collections.defaultdict(list)

    PATH = connectivitydir
    for r in os.listdir(PATH):
        if ".h5" not in r: continue
        sig_id = r.split(".h5")[0]
        pert_id = siginfo[sig_id]
        if pert_id in pertid_inchikey:
            ik = pertid_inchikey[pert_id]
            inchikey_sigid[ik] += [sig_id]

    return inchikey_sigid,inchikey_inchi


def do_ik_matrices(inchikey_sigid,connectivitydir):

    # Be careful!!! As it is now, it is a multiprocess.

    def get_summary(v):
        Qhi = np.percentile(v, 66)
        Qlo = np.percentile(v, 33)
        if np.abs(Qhi) > np.abs(Qlo):
            return Qhi
        else:
            return Qlo

    PATH = connectivitydir
    # New version, across CORE cell lines and TOUCHSTONE signatures.

    # This will take a while...

    with open("%s/signatures.tsv" % PATH, "r") as f:
        signatures = [l.rstrip("\n") for l in f]

        
    cols   = sorted(set(siginfo[s] for s in signatures))
    cols_d = dict((cols[i], i) for i in xrange(len(cols)))

    p = Pool()

    pbar = total = len(inchikey_sigid) / p._processes

    def parse_results(ik):
        v = inchikey_sigid[ik]
        neses = collections.defaultdict(list)
        for sigid in v:
            with h5py.File("%s/%s.h5" % (PATH, sigid), "r") as hf:
                nes = hf["nes"][:]
            for i in xrange(len(signatures)):
                neses[(sigid, siginfo[signatures[i]])] += [nes[i]]
        neses  = dict((x, get_summary(y)) for x,y in neses.iteritems())
        rows   = sorted(set([k[0] for k in neses.keys()]))
        rows_d = dict((rows[i], i) for i in xrange(len(rows)))
        X = np.zeros((len(rows), len(cols))).astype(np.int16)
        for x,y in neses.iteritems():
            i = rows_d[x[0]]
            j = cols_d[x[1]]
            X[i,j] = y
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "w") as hf:
            hf.create_dataset("X", data = X)
            hf.create_dataset("rows", data = rows)
        pbar.update(1)

    p.map(parse_results, inchikey_sigid.keys())


def do_consensus():

    inchikeys = [ik.split(".h5")[0] for ik in os.listdir(ik_matrices)]

    def consensus_signature(ik):
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        return [np.int16(get_summary(X[:,j])) for j in xrange(X.shape[1])] # It could be max, min...

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data = inchikeys)
        hf.create_dataset("X", data = X)

    return X, inchikeys


def process():

    def whiten(X):
        
        Xw = np.zeros(X.shape)
        
        
        
        for j in xrange(X.shape[1]):
            V = X[:,j]
            V = rankdata(V, "ordinal")
            gauss = g.Gaussianize(strategy = "brute")
            gauss.fit(V)
            V = gauss.transform(V)
            Xw[:,j] = np.ravel(V)
        
        return Xw

    Xw = whiten(X)

    def cutoffs(X):
        return [np.percentile(X[:,j], 99) for j in xrange(X.shape[1])]
            
    cuts = cutoffs(X)

    Xcut = []
    for j in xrange(len(cuts)):
        c = cuts[j]
        v = np.zeros(X.shape[0])
        v[X[:,j] > c] = 1
        Xcut += [v]
        
    Xcut = np.array(Xcut).T

    return Xcut


def insert_to_database(Xcut, inchikeys,inchikey_inchi):

    inchikey_raw = {}
    for i in xrange(len(inchikeys)):
        ik = inchikeys[i]
        if np.sum(Xcut[i,:]) < 5: continue
        idxs = np.where(Xcut[i,:] == 1)[0]
        inchikey_raw[ik] = ",".join(["%d(1)" % x for x in idxs])

    todos = Psql.insert_structures(inchikey_inchi, dbname)
    for ik in todos:
        draw(ik,inchikey_inchi[ik])
    Psql.insert_raw(transcript, inchikey_raw,dbname)


# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname

    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    global mini_sig_info_file,lincs_molrepo,ik_matrices,consensus
    
    signaturesdir = checkercfg.getDirectory( "signatures" )
    
    tmpdir = checkercfg.getDirectory( "temp" )
    

    if os.path.exists(tmpdir) == False:
        c = os.makedirs(tmpdir)

    if os.path.exists(signaturesdir) == False:
        c = os.makedirs(signaturesdir)
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    mini_sig_info_file = os.path.join(signaturesdir,'mini_sig_info.tsv')
    lincs_molrepo = os.path.join(checkercfg.getDirectory( "molRepo" ),"lincs.tsv")
    logsFiledir = checkercfg.getDirectory( "logs" )
    consensus = os.path.join(signaturesdir,"consensus.h5")

    log = logSystem(sys.stdout)
    
    
    
    log.info(  "Parsing")
    parse_level(downloadsdir)
    
    WD = os.path.dirname(os.path.realpath(__file__))
    
    ik_matrices = os.path.join(signaturesdir,'ik_matrices')


    if os.path.exists(ik_matrices) == False:
        c = os.makedirs(ik_matrices)
        
    connectivitydir = os.path.join(signaturesdir,'connectivity')

    if os.path.exists(connectivitydir) == False:
        c = os.makedirs(connectivitydir)
        
    os.chdir(connectivitydir)
    connectivity_script = WD + "/connectivity.py"

    log.info( "Getting signature files...")
    
    granularity = 10
    
    cp_sigs = set()
    with open(mini_sig_info_file) as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if l[2] == "trt_cp":
                cp_sigs.update([l[0]])
    
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    
    S = 0
    with open(filename, "w") as f:
        for l in os.listdir(signaturesdir):
            if ".h5" not in l: continue
            sig = l.split(".h5")[0]
            #if sig not in cp_sigs: continue
            if sig in cp_sigs: continue
            f.write(sig + "\n")
            S += 1
            print sig
    
    t = math.ceil(float(S)/granularity)
    jobName = 'connectivity'
    
    if os.path.exists(os.path.join(connectivitydir,jobName+'.ready')) == False:
    
        logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
        scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +connectivity_script + ' \$i ' + mini_sig_info_file + " " + signaturesdir + ' ' + os.path.join(downloadsdir,"GSE92742_Broad_LINCS_pert_info.txt")
    
        cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':filename,
                                          'COMMAND':scriptFile}
    
        
        execAndCheck(cmdStr,log)
    
        log.info( " - Launching the job %s on the cluster " % (jobName) )
        cmdStr = SUBMITJOBANDREADY+" "+tmpdir+" "+jobName+" "+logFilename
        execAndCheck(cmdStr,log)
    
    
        checkJobResultsForErrors(tmpdir,jobName,log)    


    log.info(  "Reading L1000")
    inchikey_sigid,inchikey_inchi = read_l1000(connectivitydir)

    log.info(  "Doing ik_matrices")
    do_ik_matrices(inchikey_sigid,connectivitydir)

    log.info(  "Doing consensus")
    X, inchikeys = do_consensus()

    log.info(  "Process output")
    Xcut = process()

    log.info(  "Insert to database")
    insert_to_database(Xcut,inchikeys,inchikey_inchi)


if __name__ == '__main__':
    main()