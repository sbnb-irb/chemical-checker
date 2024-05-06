import os
import time
import numpy as np

from chemicalchecker.core.chemcheck import ChemicalChecker
from chemicalchecker.database.molecule import Molecule

from e3fp import pipeline
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys, ChemicalFeatures, QED
from python_utilities.parallel import Parallelizer

def a1_fprints_from_inchi(inchi, inchikey, dense=True):
    print( 'a1', inchi, inchikey )
    
    nBits = 2048
    radius = 2
        
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            result = mfpgen.GetFingerprint(mol)
    if not result:
        result = {
            "inchikey": inchikey,
            "raw": result
        }
    else:
        s = np.array(result)
        if( dense ):
            s = ",".join( "%d" % s for s in sorted( [x for x in result.GetOnBits()] ) )
        
        result = {
            "inchikey": inchikey,
            "raw": s
        }
    return result

def a2_fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
    print( 'a2', inchi, inchikey )
    
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11:
                result = None
            else:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                result = pipeline.fprints_from_smiles( smiles, inchikey, confgen_params, fprint_params, save )
    if not result:
        result = {
            "inchikey": inchikey,
            "raw": result
        }
    else:
        s = ",".join([str(x) for fp in result for x in fp.indices])
        result = {
            "inchikey": inchikey,
            "raw": s
        }
    return result

def a3_fprints_from_inchi(inchi, inchikey):
    print( 'a3', inchi, inchikey)

    nBits = 1024
    radius = 2
        
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                if not Chem.MolToSmiles(core, isomericSmiles=True):
                    core = mol
                fw = MurckoScaffold.MakeScaffoldGeneric(core)
                info = {}
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
                c_fp = mfpgen.GetFingerprint( core ).GetOnBits()
                f_fp = mfpgen.GetFingerprint( fw ).GetOnBits()
                
                fp = ["c%d" % x for x in c_fp] + ["f%d" % y for y in f_fp]
                result = ",".join(fp)
            except Exception:
                result = None
    result = {
        "inchikey": inchikey,
        "raw": result
    }
    return result

def a4_fprints_from_inchi(inchi, inchikey):
    print( 'a4', inchi, inchikey)
        
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                result = ",".join( "%d" % s for s in sorted( [x for x in fp.GetOnBits()] ) )
            except Exception:
                result = None
    result = {
        "inchikey": inchikey,
        "raw": result
    }
    return result

def a5_fprints_from_inchi(inchi, inchikey, alerts_chembl = None):
    print( 'a5', inchi, inchikey)
        
    result = None
    if(inchi != None):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]
        if(mol != None):
            try:
                P = {}
                P['ringaliph'] = Descriptors.NumAliphaticRings(mol)
                P['mr'] = Descriptors.MolMR(mol)
                P['heavy'] = Descriptors.HeavyAtomCount(mol)
                P['hetero'] = Descriptors.NumHeteroatoms(mol)
                P['rings'] = Descriptors.RingCount(mol)

                props = QED.properties(mol)
                P['mw'] = props[0]
                P['alogp'] = props[1]
                P['hba'] = props[2]
                P['hbd'] = props[3]
                P['psa'] = props[4]
                P['rotb'] = props[5]
                P['ringarom'] = props[6]
                P['alerts_qed'] = props[7]
                P['qed'] = QED.qed(mol)

                # Ro5
                ro5 = 0
                if P['hbd'] > 5:
                    ro5 += 1
                if P['hba'] > 10:
                    ro5 += 1
                if P['mw'] >= 500:
                    ro5 += 1
                if P['alogp'] > 5:
                    ro5 += 1
                P['ro5'] = ro5

                # Ro3
                ro3 = 0
                if P['hbd'] > 3:
                    ro3 += 1
                if P['hba'] > 3:
                    ro3 += 1
                if P['rotb'] > 3:
                    ro3 += 1
                if P['mw'] >= 300:
                    ro3 += 1
                if P['alogp'] > 3:
                    ro3 += 1
                P['ro3'] = ro3

                # Structural alerts from Chembl
                P['alerts_chembl'] = len( set( [ int(f.GetType()) for f in alerts_chembl.GetFeaturesForMol(mol) ] ) )
                raw = "mw(%.2f),heavy(%d),hetero(%d),rings(%d),ringaliph(%d),ringarom(%d),alogp(%.3f),mr(%.3f)" + \
                ",hba(%d),hbd(%d),psa(%.3f),rotb(%d),alerts_qed(%d),alerts_chembl(%d),ro5(%d),ro3(%d),qed(%.3f)"
                result = raw % (P['mw'], P['heavy'], P['hetero'],
                            P['rings'], P['ringaliph'], P['ringarom'],
                            P['alogp'], P['mr'], P['hba'], P['hbd'], P['psa'],
                            P['rotb'], P['alerts_qed'], P['alerts_chembl'],
                            P['ro5'], P['ro3'], P['qed'])
            except Exception:
                result = None
    result = {
        "inchikey": inchikey,
        "raw": result
    }
    return result

# input
cc = ChemicalChecker('/aloy/web_checker/package_2024_update/')
mols = cc.get_signature('sign0', 'full', 'B5.001')
keys = mols.keys

n = 5000
ckeys = list(keys)[:50000] 
#ckeys = list(keys)
ckeys = list(keys)[:n] 
inchikey_inchi = Molecule.get_inchikey_inchi_mapping(ckeys)
inchi_iter = ((inchi, key) for key, inchi in inchikey_inchi.items())

# Treating parameters
kwargs = {}
kwargs['a1'] = { "dense": True }

root_a2 = '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/chemicalchecker/util/parser/data/defaults.cfg'
params_a2 = pipeline.params_to_dicts( root_a2 )
kwargs['a2'] = { "confgen_params": params_a2[0], "fprint_params": params_a2[1] }

kwargs['a3'] = {  }
kwargs['a4'] = {  }

root_a5 = '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/chemicalchecker/util/parser/data/structural_alerts.fdef'
alerts_chembl = ChemicalFeatures.BuildFeatureFactory( root_a5 )
kwargs['a5'] = { "alerts_chembl": alerts_chembl }
    
flog = f"log_{n}.txt"
f = open( flog, 'w')
f.close()

# Testing parallel processing
procs = int( os.environ.get("OMP_NUM_THREADS", 8) )
parallelizer = Parallelizer(parallel_mode="processes", num_proc=procs)
for i in range(1,6):
    fprints_list=[]
    inchi_iter = ((inchi, key) for key, inchi in inchikey_inchi.items())
    
    print( f"--- [Parallel] Testing fingerprints for A{i} ---" )
    start_time = time.time()
    fprints_list = parallelizer.run( eval(f"a{i}_fprints_from_inchi"), inchi_iter, kwargs = kwargs[f"a{i}"] ) 
    diff = time.time() - start_time
    print("\tContent Preview: ", fprints_list[:3])
    print("\tLength: ", len(fprints_list))
    print("\tTime: %s seconds" % (diff) )
    
    with open( flog, 'a') as f:
        f.write( f"\n--- [Parallel] Testing fingerprints for A{i} ---\n" )
        f.write( f"\tLength: { len(fprints_list) }\n" )
        f.write( f"\tTime: { diff } seconds\n" )
        
# Testing sequential processing
for n in range(1,6):
    fprints_list=[]
    inchi_iter = ((inchi, key) for key, inchi in inchikey_inchi.items())
    
    print( f"--- [Sequential] Testing fingerprints for A{n} ---" )
    start_time = time.time()
    for i in inchi_iter:
        ag = i
        kw = kwargs[f"a{n}"]
        fprints_list.append( eval(f"a{n}_fprints_from_inchi")(*ag, **kw ) )
    diff = time.time() - start_time
    print("\tContent Preview: ", fprints_list[:3])
    print("\tLength: ", len(fprints_list))
    print("\tTime: %s seconds" % (diff) )
    
    with open( flog, 'a') as f:
        f.write( f"--- [Sequential] Testing fingerprints for A{n} ---\n" )
        f.write( f"\tLength: { len(fprints_list) }\n" )
        f.write( f"\tTime: { diff } seconds\n" )

