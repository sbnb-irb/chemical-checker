# Imports

from tqdm import tqdm
import sys, os
sys.path.append(os.path.join(sys.path[0], "../../dbutils/"))
import Psql
from rdkit.Chem import AllChem as Chem

# Variables

root = os.path.dirname(os.path.realpath(__file__))

# Functions

def insert_raw(inchikey_raw, table):
    for c in Psql.chunker(inchikey_raw.keys(), 1000):
        s = ",".join(["('%s', '%s')" % (k, inchikey_raw[k]) for k in c])
        Psql.query("INSERT INTO %s (inchikey, raw) VALUES %s ON CONFLICT DO NOTHING" % (table, s), Psql.mosaic)

def todos(inchikey_inchi, table):
    return [ik for ik in Psql.not_yet_in_table(inchikey_inchi.keys(), table)]


# Main functions

def fp2d(inchikey_inchi):
    table = "fp2d"
    iks = todos(inchikey_inchi, table)
    nBits  = 2048
    radius = 2
    inchikey_raw = {}
    for k in tqdm(iks):
        v = inchikey_inchi[k]
        mol = Chem.rdinchi.InchiToMol(v)[0]
        info = {}
        fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=info)
        dense = ",".join("%d" % s for s in sorted([x for x in fp.GetOnBits()]))
        inchikey_raw[k] = dense
    insert_raw(inchikey_raw, table)


def fp3d(inchikey_inchi):
    from e3fp import pipeline
    from rdkit import Chem as sChem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from timeout import timeout
    
    @timeout(100)
    def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
        mol = Chem.rdinchi.InchiToMol(inchi)[0]

        if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11: return None

        smiles = Chem.MolToSmiles(mol)
        return pipeline.fprints_from_smiles(smiles, inchikey, confgen_params, fprint_params, save)

    table = "fp3d"
    iks = todos(inchikey_inchi, table)
    
    # Load parameters
    params = pipeline.params_to_dicts(root+"/db/defaults.cfg")

    # Calculate fingerprints
    for k in tqdm(iks):
        v = inchikey_inchi[k]
        try:
            fps = fprints_from_inchi(v, k, params[0], params[1])
        except:
            continue
        if not fps: continue
        s = ",".join([str(x) for fp in fps for x in fp.indices])
        Psql.query("INSERT INTO fp3d (inchikey, raw) VALUES ('%s', '%s') ON CONFLICT DO NOTHING" % (k, s), Psql.mosaic)  


def scaffolds(inchikey_inchi):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    table = "scaffolds"
    iks = todos(inchikey_inchi, table)
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
    for k in tqdm(iks):
        v = inchikey_inchi[k]
        mol  = Chem.rdinchi.InchiToMol(v)[0]
        try:
            dense = murcko_scaffold(mol)
        except:
            dense = None
        if not dense: continue
        inchikey_raw[k] = dense
    insert_raw(inchikey_raw, table)


def subskeys(inchikey_inchi):
    from rdkit.Chem import MACCSkeys
    table = "subskeys"
    iks = todos(inchikey_inchi, table)
    inchikey_raw = {}
    for k in tqdm(iks):
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

    alerts_chembl = ChemicalFeatures.BuildFeatureFactory(root+"/db/structural_alerts.fdef") # Find how to create this file in netscreens/physchem

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
        Psql.query("INSERT INTO physchem (inchikey, mw, heavy, hetero, rings, ringaliph, ringarom, alogp, mr, hba, hbd, psa, rotb, alerts_qed, alerts_chembl, ro5, ro3, qed) VALUES %s ON CONFLICT DO NOTHING" % S, Psql.mosaic)

    for c in tqdm(Psql.chunker(iks, 1000)):
        insert_descriptor(c)

    #p = Pool()

    #p.map(insert_descriptor, [c for c in Psql.chunker(iks, 1000)])

