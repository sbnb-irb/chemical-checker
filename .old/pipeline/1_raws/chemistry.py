#!/miniconda/bin/python

# Imports

import collections
import pybel
import subprocess
import os, sys
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck, draw,checkJobResultsForErrors,compressJobResults
import uuid
import math
import Psql
from rdkit.Chem import AllChem as Chem
import checkerconfig

# Variables

log = ''
dbname = ''

root = os.path.dirname(os.path.realpath(__file__))

SUBMITJOBANDREADY = os.path.join(sys.path[0],'../../src/utils/submitJobOnClusterAndReady.py')

# Functions

def todos(inchikey_inchi, table):
    return [ik for ik in Psql.not_yet_in_table(inchikey_inchi.keys(), table,dbname)]

def insert_raw(inchikey_raw, table):
    for c in Psql.chunker(inchikey_raw.keys(), 1000):
        s = ",".join(["('%s', '%s')" % (k, inchikey_raw[k]) for k in c])
        Psql.query("INSERT INTO %s (inchikey, raw) VALUES %s ON CONFLICT DO NOTHING" % (table, s), dbname)
        
def fetch_inchikeys(table):
        return [r[0] for r in Psql.qstring("SELECT inchikey FROM %s" % table, dbname)]

def fetch_inchies(iks):
    for c in Psql.chunker(iks, 1000):
        s = "(" + ",".join(["'%s'" % x for x in c]) + ")"
        #s = "(" + ",".join(c) + ")"
        cmd = "SELECT inchikey,inchi from structure where inchikey in  " + s
        for r in Psql.qstring(cmd, dbname):
            yield r


def get_inchikeys(tables,repos_dir):
    
    mymols = set()

    for t in tables: mymols.update(fetch_inchikeys(t))
    
    print len(mymols)
    inchikey_inchi = {}
    inchikey_inchi_final = {}
    for repo in os.listdir(repos_dir):
        f = open(os.path.join(repos_dir,repo), "r")
    
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            
            inchikey_inchi[l[2]] = l[3]

                        
    
    inchikey_inchi_final = {ik:inchikey_inchi[ik] for ik in mymols if ik in inchikey_inchi  }
   

    print len(inchikey_inchi_final.keys())
    
    return inchikey_inchi_final


def fp2d(inchikey_inchi):
    table = "fp2d"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    nBits  = 2048
    radius = 2
    inchikey_raw = {}
    for k in iks:
        v = inchikey_inchi[k]
        mol = Chem.rdinchi.InchiToMol(v)[0]
        info = {}
        #print mol
        fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=info)
        dense = ",".join("%d" % s for s in sorted([x for x in fp.GetOnBits()]))
        inchikey_raw[k] = dense
    
    
    
    insert_raw(inchikey_raw, table)


def fp3d(inchikey_inchi):
    from e3fp import pipeline
    from rdkit import Chem as sChem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    import time
    from timeout import timeout
    import timeout_decorator

    
    @timeout_decorator.timeout(100, use_signals=False)
    #@timeout(100)
    def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]

        if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11: return None

        smiles = Chem.MolToSmiles(mol)
        return pipeline.fprints_from_smiles(smiles, inchikey, confgen_params, fprint_params, save)

    table = "fp3d"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    # Load parameters
    params = pipeline.params_to_dicts(root+"/../files/defaults.cfg")

    
    # Calculate fingerprints
    for k in iks:
        v = inchikey_inchi[k]
        try:
            fps = fprints_from_inchi(v, k, params[0], params[1])
        except Exception as inst:
            print 'Timeout inchikey: ' + k
            fps = None
        if not fps: 
            Psql.query("INSERT INTO fp3d (inchikey, raw) VALUES ('%s', NULL) ON CONFLICT DO NOTHING" % (k), dbname)  
        else:
            s = ",".join([str(x) for fp in fps for x in fp.indices])
            Psql.query("INSERT INTO fp3d (inchikey, raw) VALUES ('%s', '%s') ON CONFLICT DO NOTHING" % (k, s), dbname)  

def fp3d_multi(inchikey_inchi):
    

    table = "fp3d"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    # Load parameters
    
    

    if os.path.exists(tmpdir) == False:
        c = os.makedirs(tmpdir)
    
    filename = os.path.join(tmpdir , str(uuid.uuid4()))
    
    S = 0
    granularity = 40
    with open(filename, "w") as f:
        for ik in iks:
          
            f.write(ik + "-----"+ inchikey_inchi[ik] +  "\n")
            S += 1
            #print sig
    
    t = math.ceil(float(S)/granularity)
    fp3d_dir = os.path.join(tmpdir,table)
    
    if os.path.exists(fp3d_dir) == False:
        c = os.makedirs(fp3d_dir)
        
    os.chdir(fp3d_dir)

    jobName = 'chem-fp3d'
    
    if os.path.exists(os.path.join(fp3d_dir,jobName+'.ready')) == False:
    
        logFilename = os.path.join(logsFiledir,jobName+".qsub")
    
        scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + ' python ' + root + '/fp3d-fingerprint.py ' + dbname + ' \$i ' 
    
        cmdStr = checkerconfig.SETUPARRAYJOB % { 'JOB_NAME':jobName, 'NUM_TASKS':t,
                                          'TASKS_LIST':filename,
                                          'COMMAND':scriptFile}
    
        
        execAndCheck(cmdStr,log)
    
        log.info( " - Launching the job %s on the cluster " % (jobName) )
        cmdStr = SUBMITJOBANDREADY+" "+fp3d_dir+" "+jobName+" "+logFilename
        execAndCheck(cmdStr,log)
    
    
        #checkJobResultsForErrors(fp3d_dir,jobName,log)    
        compressJobResults(fp3d_dir,jobName,['tasks'],log)

  

def scaffolds(inchikey_inchi):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    table = "scaffolds"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    nBits = 1024
    radius = 2
    def murcko_scaffold(mol):
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if not Chem.MolToSmiles(core): core = mol
        fw = MurckoScaffold.MakeScaffoldGeneric(core)
        info = {}
        c_fp = Chem.GetMorganFingerprintAsBitVect(core, radius, nBits=nBits, bitInfo=info).GetOnBits()
        f_fp = Chem.GetMorganFingerprintAsBitVect(fw, radius, nBits=nBits, bitInfo=info).GetOnBits()
        fp = ["c%d" % x for x in c_fp] + ["f%d" % y for y in f_fp]
        return ",".join(fp)
    inchikey_raw = {}
    
        
    for k in iks:
        v = inchikey_inchi[k]
        mol  = Chem.rdinchi.InchiToMol(v)[0]
        try:
            dense = murcko_scaffold(mol)
        except:
            dense = None
        if not dense: 
            Psql.query("INSERT INTO scaffolds (inchikey, raw) VALUES ('%s', NULL) ON CONFLICT DO NOTHING" % (k), dbname)
        else:
            inchikey_raw[k] = dense
    insert_raw(inchikey_raw, table)


def subskeys(inchikey_inchi):
    from rdkit.Chem import MACCSkeys
    table = "subskeys"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    inchikey_raw = {}
    for k in iks:
        v = inchikey_inchi[k]
        mol  = Chem.rdinchi.InchiToMol(v)[0]
        info = {}
        fp = MACCSkeys.GenMACCSKeys(mol)
        dense = ",".join("%d" % s for s in sorted([x for x in fp.GetOnBits()]))
        inchikey_raw[k] = dense
        
    
    insert_raw(inchikey_raw, table)


def physchem(inchikey_inchi):

    from silicos_it.descriptors import qed
    from rdkit.Chem import Descriptors
    from rdkit.Chem import ChemicalFeatures
    #from multiprocessing import Pool
    table = "physchem"
    iks = todos(inchikey_inchi, table)
    log.info( "Found " + str(len(iks)) + " molecules missing")
    alerts_chembl = ChemicalFeatures.BuildFeatureFactory(root+"/../files/structural_alerts.fdef") # Find how to create this file in netscreens/physchem

    def descriptors(mol):
        P = {}

        P['ringaliph'] = Descriptors.NumAliphaticRings(mol)
        P['mr'] = Descriptors.MolMR(mol)
        P['heavy'] = Descriptors.HeavyAtomCount(mol)
        P['hetero'] = Descriptors.NumHeteroatoms(mol)
        P['rings'] = Descriptors.RingCount(mol)

        props = qed.properties(mol)
        P['mw'] = props[0]
        P['alogp'] = props[1]
        P['hba'] = props[2]
        P['hbd'] = props[3]
        P['psa'] = props[4]
        P['rotb'] = props[5]
        P['ringarom'] = props[6]
        P['alerts_qed'] = props[7]
        P['qed'] = qed.qed([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], props, True)

        # Ro5
        ro5 = 0
        if P['hbd'] > 5  : ro5 += 1
        if P['hba'] > 10 : ro5 += 1
        if P['mw'] >= 500: ro5 += 1
        if P['alogp'] > 5: ro5 += 1
        P['ro5'] = ro5

        # Ro3
        ro3 = 0
        if P['hbd'] > 3  : ro3 += 1
        if P['hba'] > 3  : ro3 += 1
        if P['rotb'] > 3 : ro3 += 1
        if P['mw'] >= 300: ro3 += 1
        if P['alogp'] > 3: ro3 += 1
        P['ro3'] = ro3

        # Structural alerts from Chembl
        P['alerts_chembl'] = len(set([int(feat.GetType()) for feat in alerts_chembl.GetFeaturesForMol(mol)]))
        return P


    def insert_descriptor(c):
        S = []
        for k in c:
            v = inchikey_inchi[k]
            mol = Chem.rdinchi.InchiToMol(v)[0]
            P = descriptors(mol)
            s = "('%s',%.2f,%d,%d,%d,%d,%d,%.3f,%.3f,%d,%d,%.3f,%d,%d,%d,%d,%d,%.3f)" % (k, P['mw'], P['heavy'], P['hetero'],
                                                                                         P['rings'], P['ringaliph'], P['ringarom'],
                                                                                         P['alogp'], P['mr'], P['hba'], P['hbd'], P['psa'],
                                                                                         P['rotb'], P['alerts_qed'], P['alerts_chembl'],
                                                                                         P['ro5'], P['ro3'], P['qed'])
            S += [s]
        S = ",".join(S)
        Psql.query("INSERT INTO physchem (inchikey, mw, heavy, hetero, rings, ringaliph, ringarom, alogp, mr, hba, hbd, psa, rotb, alerts_qed, alerts_chembl, ro5, ro3, qed) VALUES %s ON CONFLICT DO NOTHING" % S, dbname)

    
    for c in Psql.chunker(iks, 1000):
        insert_descriptor(c)



# Main

def main():
    
    import argparse
    
    if len(sys.argv) != 2:
        sys.exit(1)
  
    configFilename = sys.argv[1]

    checkercfg = checkerconfig.checkerConf( configFilename)  
    global dbname,log,tmpdir,logsFiledir
    
    dbname = checkerconfig.dbname + "_" + checkercfg.getVariable("General",'release')
    
    downloadsdir = checkercfg.getDirectory( "downloads" )
    tmpdir = checkercfg.getDirectory( "temp" )
    
    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)
    
    dirName = os.path.abspath(sys.argv[0]).split("/")[-2]
    readyFiledir = checkercfg.getDirectory( "ready" )

    all_tables = checkercfg.getTableList("all")
    
    all_chemistry = checkercfg.getTableList("chemistry")
    tables = list(set(all_tables) - set(all_chemistry))
    
    print tables

    log.info( "Get list of inchikeys...")
    inchikey_inchi = get_inchikeys(tables,checkercfg.getDirectory( "molRepo" ))
    
    readyFilename = os.path.join(readyFiledir,dirName+"_"+"fp2d.ready")
    
    if os.path.exists(readyFilename) == False:

        log.info( "Filling fp2d table.")
    
        fp2d(inchikey_inchi)
        
        cmdStr = "touch "+readyFilename
        subprocess.call(cmdStr,shell=True)
    
    readyFilename = os.path.join(readyFiledir,dirName+"_"+"fp3d.ready")
    
    if os.path.exists(readyFilename) == False:
    
        log.info( "Filling fp3d table.")
    
        fp3d_multi(inchikey_inchi)
        
        cmdStr = "touch "+readyFilename
        subprocess.call(cmdStr,shell=True)
    
    readyFilename = os.path.join(readyFiledir,dirName+"_"+"scaffolds.ready")
        
    if os.path.exists(readyFilename) == False:

        log.info( "Filling scaffolds table.") 
    
        scaffolds(inchikey_inchi)
        
        cmdStr = "touch "+readyFilename
        subprocess.call(cmdStr,shell=True)
    
    
    readyFilename = os.path.join(readyFiledir,dirName+"_"+"subskeys.ready")
    
    if os.path.exists(readyFilename) == False:
    
        log.info( "Filling subskeys table.")
    
        subskeys(inchikey_inchi)
        
        cmdStr = "touch "+readyFilename
        subprocess.call(cmdStr,shell=True)
    
    readyFilename = os.path.join(readyFiledir,dirName+"_"+"physchem.ready")
    
    if os.path.exists(readyFilename) == False:
    
        log.info( "Filling physchem table.")
    
        physchem(inchikey_inchi)
        cmdStr = "touch "+readyFilename
        subprocess.call(cmdStr,shell=True)
    
  


if __name__ == "__main__":
    main()