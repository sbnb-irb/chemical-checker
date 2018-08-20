#!/miniconda/bin/python

# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw,checkJobResultsForErrors,compressJobResults
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

    return inchikey_sigid,inchikey_inchi,siginfo


def do_ik_matrices(inchikey_sigid,connectivitydir,siginfo):

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

    #p = Pool()

    #pbar = total = len(inchikey_sigid) / p._processes

    print  len(inchikey_sigid.keys())
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
        #pbar.update(1)

    for key in inchikey_sigid.keys():
        parse_results(key)
    #p.map(parse_results, inchikey_sigid.keys())

def get_summary(v):
        Qhi = np.percentile(v, 66)
        Qlo = np.percentile(v, 33)
        if np.abs(Qhi) > np.abs(Qlo):
            return Qhi
        else:
            return Qlo
        
        
def do_consensus():

    inchikeys = [ ik.split(".h5")[0] for ik in os.listdir(ik_matrices) if ik.endswith(".h5")]

    def consensus_signature(ik):
        with h5py.File("%s/%s.h5" % (ik_matrices, ik), "r") as hf:
            X = hf["X"][:]
        return [np.int16(get_summary(X[:,j])) for j in xrange(X.shape[1])] # It could be max, min...

    X = np.array([consensus_signature(ik) for ik in inchikeys])

    with h5py.File(consensus, "w") as hf:
        hf.create_dataset("inchikeys", data = inchikeys)
        hf.create_dataset("X", data = X)

    return X, inchikeys


def process(X):

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
    Psql.insert_raw(table, inchikey_raw,dbname)


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
        
   
    
    WD = os.path.dirname(os.path.realpath(__file__))
    
    ik_matrices = os.path.join(signaturesdir,'ik_matrices')


    if os.path.exists(ik_matrices) == False:
        c = os.makedirs(ik_matrices)
        
    connectivitydir = os.path.join(signaturesdir,'connectivity')

    if os.path.exists(connectivitydir) == False:
        c = os.makedirs(connectivitydir)
        
    os.chdir(connectivitydir)
    connectivity_script = WD + "/connectivity.py"
    
    ikmatrices_script = WD + "/do_ik_matrices.py"

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
            if sig not in cp_sigs: continue
            #if sig in cp_sigs: continue
            f.write(sig + "\n")
            S += 1
            #print sig
    
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
        cmdStr = SUBMITJOBANDREADY+" "+connectivitydir+" "+jobName+" "+logFilename
        execAndCheck(cmdStr,log)
    
    
        checkJobResultsForErrors(connectivitydir,jobName,log)    
        compressJobResults(connectivitydir,jobName,['tasks'],log)


    log.info(  "Reading L1000")
    inchikey_sigid,inchikey_inchi,siginfo = read_l1000(connectivitydir)
    
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    
    S = 0
    granularity = 40
    with open(filename, "w") as f:
        for key in inchikey_sigid.keys():
           
            f.write(key + "\n")
            S += 1
            
    jobName = 'ikmatrices'
    
    t = math.ceil(float(S)/granularity)
    os.chdir(ik_matrices)
    log.info(  "Doing ik matrices")
    
    if os.path.exists(os.path.join(ik_matrices,jobName+'.ready')) == False:
    
        logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
        scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' +ikmatrices_script + ' \$i ' + mini_sig_info_file + " " + connectivitydir + ' ' + ik_matrices + ' ' + lincs_molrepo
    
        cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':filename,
                                          'COMMAND':scriptFile}
    
        
        execAndCheck(cmdStr,log)
    
        log.info( " - Launching the job %s on the cluster " % (jobName) )
        cmdStr = SUBMITJOBANDREADY+" "+ik_matrices+" "+jobName+" "+logFilename
        execAndCheck(cmdStr,log)
    
    
        checkJobResultsForErrors(ik_matrices,jobName,log)    
        compressJobResults(ik_matrices,jobName,['tasks'],log)

        

    log.info(  "Doing consensus")
    X, inchikeys = do_consensus()

    log.info(  "Process output")
    Xcut = process(X)

    log.info(  "Insert to database")
    insert_to_database(Xcut,inchikeys,inchikey_inchi)


if __name__ == '__main__':
    main()
