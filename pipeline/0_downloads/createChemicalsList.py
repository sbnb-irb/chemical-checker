#!/miniconda/bin/python

import os,sys,string,commands
import urllib2

sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))


import checkerconfig

def get_PubChemID_from_ChemicalID(chemicalid):
    try:
        return urllib2.urlopen('http://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/Comparative%20Toxicogenomics%20Database/' + chemicalid + '/cids/TXT/').read().rstrip()
    except:
        return None


def get_Smiles_from_PubChemID(cid):
    try:
        return urllib2.urlopen('http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/property/CanonicalSMILES/TXT/' % cid).read().rstrip()
    except:
        return None


def get_Smiles_from_ChemicalName(chem_name):
    try:
        chem_name = urllib2.quote(chem_name)
        return urllib2.urlopen('http://cactus.nci.nih.gov/chemical/structure/%s/smiles' % chem_name).read().rstrip()
    except:
        return None

def chemical_to_smiles(chemid, cn):
    cid = get_PubChemID_from_ChemicalID(chemid)
    smiles = get_Smiles_from_PubChemID(cid)
    if smiles is None:
        smiles = get_Smiles_from_ChemicalName(cn)
    return smiles


if len(sys.argv) != 2:
    usage(sys.argv[0])
    sys.exit(1)
  
configFilename = sys.argv[1]

checkercfg = checkerconfig.checkerConf(configFilename )  

logsFiledir = checkercfg.getDirectory( "logs" )

log = logSystem(sys.stdout)

downloadsdir = checkercfg.getDirectory( "downloads" )

check_dir = os.path.exists(logsFiledir)


if check_dir == False:
  c = os.makedirs(logsFiledir)

check_dir = os.path.exists(downloadsdir)


if check_dir == False:
  c = os.makedirs(downloadsdir)

os.chdir(downloadsdir)

log.info( "Reading CTD Chemicals Disease")


C = set()
f = open(os.path.join(downloadsdir,"CTD_chemicals_diseases.tsv"), "r")
for l in f:
    if l[0] == "#": continue
    l = l.rstrip("\n").split("\t")
    cn = l[0]
    chemid = l[1]
    if not l[5]: continue
    C.update([(chemid, cn)])
f.close()

chem_smiles = {}

log.info( "Querying Smiles from Pubchem")

for c in C:
    if c[0] in chem_smiles: continue
    smiles = chemical_to_smiles(c[0], c[1])
    if smiles is None: continue
    chem_smiles[c[0]] = smiles

log.info( "Writing Smiles to file")

f = open(os.path.join(downloadsdir,checkerconfig.ctd_molecules_download), "w")
for k,v in chem_smiles.iteritems():
    f.write("%s\t%s\n" % (k,v))
f.close()
    
