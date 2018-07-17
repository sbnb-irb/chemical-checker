#!/miniconda/bin/python


# Imports

import sys, os
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw
import Psql
import collections
import numpy as np

import networkx as nx

import csv
import checkerconfig


# Variables

id_conversion    = "XXXX" # Metaphors - id_conversion.txt
file_9606        = "XXXX" # Metaphors - 9606.txt
human_proteome   = "XXXX" # Human proteome - human_proteome.tab
prot_allgos      = "XXXX" # ONE MUST RUN db/_prot_allgos.py BEFORE!!!!
goa_file = ''
table = "bps"

dbname = '' # 

granularity = 20


myfold = os.path.dirname(os.path.realpath(__file__))

script_file = myfold + "/prot2allgos.py"


SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')


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
    for r in tqdm(R):
        for x in r[1].split(","):
            uniprot_ac, act = x.split("(")
            act = int(act.split(")")[0])
            if uniprot_ac not in any_human: continue
            hps = any_human[uniprot_ac]
            
            for hp in hps:    
                ACTS[(r[0], hp)] += [act]
    ACTS = dict((k, np.max(v)) for k,v in ACTS.iteritems())

    any_human.clear()

    def get_allgos(uniprot_ac):
        if not os.path.exists("%s/%s.tsv" % (prot_allgos, uniprot_ac)): return None
        gos = set()
        f = open("%s/%s.tsv" % (prot_allgos, uniprot_ac), "r")
        for l in f:
            go = l.rstrip("\n")
            gos.update([go])
        f.close()
        return gos

    # Get prot and go

    prots = set([x[1] for x in ACTS.keys()])

    class_prot = {}
    for prot in prots:
        gos = get_allgos(prot)
        if not gos: continue
        class_prot[prot] = gos

    GOS = collections.defaultdict(list)
    for k,v in ACTS.iteritems():
        if k[1] not in class_prot: continue
        for go in class_prot[k[1]]:
            GOS[(k[0], go)] += [v]
    GOS = dict((k, np.max(v)) for k,v in GOS.iteritems())

    return GOS


def insert_to_database(GOS):

    inchikey_raw = collections.defaultdict(list)
    for k,v in tqdm(GOS.iteritems()):
        inchikey_raw[k[0]] += [(k[1], v)]

    inchikey_raw = dict((k, ",".join(["%s(%d)" % (x[0], x[1]) for x in v])) for k,v in tqdm(inchikey_raw.iteritems()))

    Psql.insert_raw(table, inchikey_raw,dbname)

# Main

def main():

    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,id_conversion,file_9606,human_proteome,prot_allgos
    logsFiledir = checkercfg.getDirectory( "logs" )
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    tmpdir = checkercfg.getDirectory( "temp" )
    

    if os.path.exists(tmpdir) == False:
        c = os.makedirs(tmpdir)
    
    id_conversion = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.id_conversion)
    file_9606 = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.file_9606)
    human_proteome = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.human_proteome)
    prot_allgos = os.path.join(tmpdir,'prot_allgos')
    goa_file = os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.goa_human)
    
    G = nx.DiGraph()

    f = open(os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.go_file), "r")
    S = f.read().split("[Term]")[1:]
    f.close()
    
    for chunk in S:
        s = chunk.split("\n")
        is_bp = False
        for l in s:
            if l[:3] == "id:":
                child = l.split("id: ")[1]
            if l[:29] == "namespace: biological_process":
                is_bp = True
            if l[:5] == "is_a:" and is_bp:
                parent = l.split("is_a: ")[1].split(" !")[0]
                if parent == "regulates": continue
                G.add_edge(parent, child)
    
    f = open(os.path.join(tmpdir,"/bp.tsv"), "w")
    for e in G.edges():
        f.write("%s\t%s\n" % (e[0], e[1]))
    f.close()
    
    uniprots = set()
    f = open(os.path.join(checkercfg.getDirectory( "downloads" ),checkerconfig.goa_human), "r")
    for l in f:
        if l[0] == "!": continue
        l = l.split("\t")
        if l[8] != "P": continue
        uniprots.update([l[1]])
    f.close()
    
    # Main
    
    # Getting infile
    f = open(os.path.join(tmpdir,'bpanno.tsv'), "w")
    for p in uniprots:
        f.write("%s\n" % p)
    f.close()
    
    # Get t
    t = math.ceil(float(len(uniprots))/granularity)
    
    os.chdir(tmpdir)
    
    if os.path.exists(prot_allgos) == False:
        c = os.makedirs(prot_allgos)
    
    jobName = 'bpanno'
    
    if os.path.exists(os.path.join(tmpdir,jobName+'.ready')) == False:
    
        logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
        scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' + myfold + '/prot2allgos.py \$i ' + tmpdir + " " + goa_file
    
        cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':os.path.join(tmpdir,'bpanno.tsv'),
                                          'COMMAND':scriptFile}
    
        
        execAndCheck(cmdStr,log)
    
        log.info( " - Launching the job %s on the cluster " % (jobName) )
        cmdStr = SUBMITJOBANDREADY+" "+tmpdir+" "+jobName+" "+logFilename
        execAndCheck(cmdStr,log)
    
    
        checkJobResultsForErrors(tmpdir,jobName,log)    

    log = logSystem(sys.stdout)
    log.info( "Reading human MetaPhors")
    any_human = human_metaphors()

    log.info( "Fetching binding data")
    GOS = fetch_binding(any_human)

    log.info( "Inserting to database")
    insert_to_database(GOS)


if __name__ == '__main__':
    main()

