"""Calculate data for molecules.

Each calc function here is iterating on a list of InChIKeys of standardised
molecules.
Each molecule is loaded, and properties/descriptors/features are computed.
The raw features are yielded in chunks as dictionaries.
These methods are used to populate the
:mod:`~chemicalchecker.database.calcdata`
database where the table has the same name as functions defined here.
"""
import os
import numpy as np

from chemicalchecker.util import logged, Config
from chemicalchecker.util.decorator import timeout

@logged
class DataCalculator():
    """DataCalculator class."""

    @staticmethod
    def calc_fn(function):
        """Serve a calc function."""
        try:
            return eval('DataCalculator.' + function)
        except Exception as ex:
            DataCalculator.__log.error(
                "Cannot find calculator function %s", function)
            raise ex

    @staticmethod
    def morgan_fp_r2_2048(inchikey_inchi, chunks=1000, dense=True):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        iks = inchikey_inchi.keys()
        nBits = 2048
        radius = 2
        chunk = list()
        for ik in iks:
            if ik is None:
                continue
            v = str(inchikey_inchi[ik])
            try:
                mol = Chem.rdinchi.InchiToMol(v)[0]
            except Exception as ex:
                DataCalculator.__log.debug("Skipping molecule %s" % ik)
                DataCalculator.__log.debug(str(ex))
                continue
            info = {}
            # print mol
            #fp = Chem.GetMorganFingerprintAsBitVect( mol, radius, nBits=nBits, bitInfo=info)
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = mfpgen.GetFingerprint(mol)
            if dense:
                dense = ",".join( "%d" % s for s in sorted( [x for x in fp.GetOnBits()] ) )
                result = {
                    "inchikey": ik,
                    "raw": dense
                }
            else:
                result = {
                    "inchikey": ik,
                    "raw": np.array(fp)
                }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def e3fp_3conf_1024(inchikey_inchi, chunks=1000, cores=None):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        try:
            from e3fp import pipeline
        except ImportError:
            raise ImportError("requires e3fp " +
                              "https://github.com/keiserlab/e3fp")

        root = os.path.dirname(os.path.realpath(__file__))

        params = pipeline.params_to_dicts(root + "/data/defaults.cfg")

        @timeout(100, use_signals=False)
        def fprints_from_inchi(inchi, inchikey, confgen_params={},
                               fprint_params={}, save=False):
            mol = Chem.rdinchi.InchiToMol(inchi)[0]

            if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11:
                return None

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            result = pipeline.fprints_from_smiles(
                smiles, inchikey, confgen_params, fprint_params, save)
            return result

        iks = inchikey_inchi.keys()
        chunk = list()
        for k in iks:
            if k is None:
                continue
            try:
                fps = fprints_from_inchi( str(inchikey_inchi[k]), str(k), params[0], params[1])
            except Exception:
                DataCalculator.__log.warning('Timeout inchikey: ' + k)
                fps = None
            if not fps:
                result = {
                    "inchikey": k,
                    "raw": fps
                }
            else:
                s = ",".join([str(x) for fp in fps for x in fp.indices])
                result = {
                    "inchikey": k,
                    "raw": s
                }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def e3fp_3conf_1024_parallel(inchikey_inchi, chunks=1000, cores=4):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            from python_utilities.parallel import Parallelizer
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        try:
            from e3fp import pipeline
        except ImportError:
            raise ImportError("requires e3fp " +
                              "https://github.com/keiserlab/e3fp")

        root = os.path.dirname(os.path.realpath(__file__))

        params = pipeline.params_to_dicts(root + "/data/defaults.cfg")

        @timeout(100, use_signals=False)
        def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
            try:
                result = None
                if(inchi != None):
                    mol = Chem.rdinchi.InchiToMol(inchi)[0]
                    if(mol != None):
                        if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11:
                            result = None
                        else:
                            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                            result = pipeline.fprints_from_smiles( smiles, inchikey, confgen_params, fprint_params, save )
            except Exception:
                result = None
            if not result:
                result = {
                    "inchikey": inchikey,
                    "raw": result
                }
            else:
                s = ",".join([str(x) for fp in result for x in fp.indices ])
                result = {
                    "inchikey": inchikey,
                    "raw": result
                }
            return result
        
        kwargs = {"confgen_params": params[0], "fprint_params": params[1] }
        parallelizer = Parallelizer(parallel_mode="processes", num_proc=cores)
        items = inchikey_inchi.items()
        for i in range(0, len(items), chunks):
            part = items[i:i+chunks]
            inchi_iter = ( (inchi, key) for key, inchi in part )
            chunk = parallelizer.run(fprints_from_inchi, inchi_iter, kwargs=kwargs) 
            yield chunk

    @staticmethod
    def murcko_1024_cframe_1024(inchikey_inchi, chunks=1000):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")

        iks = inchikey_inchi.keys()
        nBits = 1024
        radius = 2

        def murcko_scaffold(mol):
            core = MurckoScaffold.GetScaffoldForMol(mol)
            if not Chem.MolToSmiles(core, isomericSmiles=True):
                core = mol
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            info = {}
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            c_fp = mfpgen.GetFingerprint( core ).GetOnBits()
            f_fp = mfpgen.GetFingerprint( fw ).GetOnBits()
            
            #c_fp = Chem.GetMorganFingerprintAsBitVect( core, radius, nBits=nBits, bitInfo=info).GetOnBits()
            #f_fp = Chem.GetMorganFingerprintAsBitVect( fw, radius, nBits=nBits, bitInfo=info).GetOnBits()
            
            fp = ["c%d" % x for x in c_fp] + ["f%d" % y for y in f_fp]
            return ",".join(fp)

        chunk = list()
        for k in iks:
            if k is None:
                continue
            v = str(inchikey_inchi[k])
            try:
                mol = Chem.rdinchi.InchiToMol(v)[0]
            except Exception as ex:
                DataCalculator.__log.debug("Skipping molecule %s" % k)
                DataCalculator.__log.debug(str(ex))
                continue
            try:
                dense = murcko_scaffold(mol)
            except Exception:
                dense = None

            result = {
                "inchikey": k,
                "raw": dense
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def maccs_keys_166(inchikey_inchi, chunks=1000):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem import MACCSkeys
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")

        iks = inchikey_inchi.keys()
        chunk = list()
        for k in iks:
            if k is None:
                continue
            v = str(inchikey_inchi[k])
            try:
                mol = Chem.rdinchi.InchiToMol(v)[0]
            except Exception as ex:
                DataCalculator.__log.debug("Skipping molecule %s" % k)
                DataCalculator.__log.debug(str(ex))
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            dense = ",".join("%d" % s for s in sorted(
                [x for x in fp.GetOnBits()]))
            result = {
                "inchikey": k,
                "raw": dense
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def general_physchem_properties(inchikey_inchi, chunks=1000):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit.Chem import Descriptors, ChemicalFeatures, QED
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")

        def descriptors(mol):
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
            P['alerts_chembl'] = len(
                set([int(f.GetType()) for f
                     in alerts_chembl.GetFeaturesForMol(mol)]))
            return P

        root = os.path.dirname(os.path.realpath(__file__))

        # Find how to create this file in netscreens/physchem
        alerts_chembl = ChemicalFeatures.BuildFeatureFactory(
            root + "/data/structural_alerts.fdef")

        iks = inchikey_inchi.keys()
        chunk = list()

        for k in iks:
            if k is None:
                continue
            v = str(inchikey_inchi[k])
            try:
                mol = Chem.rdinchi.InchiToMol(v)[0]
            except Exception as ex:
                DataCalculator.__log.debug("Skipping molecule %s" % k)
                DataCalculator.__log.debug(str(ex))
                continue
            try:
                P = descriptors(mol)
                raw = "mw(%.2f),heavy(%d),hetero(%d),rings(%d),ringaliph(%d),ringarom(%d),alogp(%.3f),mr(%.3f)" + \
                ",hba(%d),hbd(%d),psa(%.3f),rotb(%d),alerts_qed(%d),alerts_chembl(%d),ro5(%d),ro3(%d),qed(%.3f)"
                raw = raw % (P['mw'], P['heavy'], P['hetero'],
                            P['rings'], P['ringaliph'], P['ringarom'],
                            P['alogp'], P['mr'], P['hba'], P['hbd'], P['psa'],
                            P['rotb'], P['alerts_qed'], P['alerts_chembl'],
                            P['ro5'], P['ro3'], P['qed'])
            except:
                raw = None
                
            result = {
                "inchikey": k,
                "raw": raw
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def chembl_target_predictions_v23_10um(inchikey_inchi, chunks=1000):
        try:
            from rdkit.Chem import AllChem as Chem
            from rdkit import DataStructs
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        import joblib
        import collections
        import numpy as np
        import pandas as pd
        import glob
        from chemicalchecker.database import Datasource
        # Variables
        max_targets = 20
        min_targets = 1
        min_proba = 0.5
        # Paths
        config = Config()
        maindb = config.DB.database
        models_path = Datasource.get("chembl_target_predictions", maindb)[0].data_path
        uniprot_mapping_path = Datasource.get("chembl_uniprot_mapping", maindb)[0].data_path
        
        hint = models_path
        res = glob.glob( hint+'/*', recursive=False)
        flag = len(list( filter( lambda x: x.endswith('.pkl'), res))) == 0
        while (flag) :
            hint+='/*'
            res = glob.glob( hint+'/*', recursive=False)
            flag = ( len(list( filter( lambda x: x.endswith('.pkl'), res))) == 0 ) and ( len(res) > 0 )
        fs = list( filter( lambda x: x.endswith('mNB_10uM_all.pkl'), res))[0]
        
        # Load models
        #morgan_nb = joblib.load( models_path + "/models_23/10uM/mNB_10uM_all.pkl" )
        morgan_nb = joblib.load( fs )
        classes = list(morgan_nb.targets)
        # Read target-to-uniprot mapping
        chembltarg2prot = collections.defaultdict(set)
        with open(uniprot_mapping_path + "/chembl_uniprot_mapping.txt", "r") as f:
            for l in f:
                if l[0] == "#":
                    continue
                l = l.rstrip("\n").split("\t")
                chembltarg2prot[l[1]].update([l[0]])
        # Fuction for target prediction

        def predict_targets(inchi):
            mol = Chem.rdinchi.InchiToMol(inchi)[0]
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = mfpgen.GetFingerprint( mol )
            #fp = Chem.GetMorganFingerprintAsBitVect( mol, 2, nBits=2048, bitInfo={})
            
            res = np.zeros(len(fp), int )
            DataStructs.ConvertToNumpyArray(fp, res)
            
            probas = list(morgan_nb.predict_proba(res.reshape(1, -1))[0])
            predictions = pd.DataFrame( zip(classes, probas), columns=['id', 'proba'])
            
            top_preds = predictions.sort_values( by='proba', ascending=False ).head(max_targets)
            top_preds = top_preds[top_preds["proba"] >= min_proba]
            prots = []
            for r in np.array(top_preds):
                if r[0] not in chembltarg2prot:
                    continue
                for p in chembltarg2prot[r[0]]:
                    prots += [(p, int(r[1] * 10))]
            if len(prots) < min_targets:
                return None
            return prots
            
        # Start iterating
        iks = inchikey_inchi.keys()
        chunk = list()
        for ik in iks:
            if ik is None:
                continue
            v = str(inchikey_inchi[ik])
            # DataCalculator.__log.info( ik)
            targs = predict_targets(v)
            if not targs:
                result = {
                    "inchikey": ik,
                    "raw": targs
                }
            else:
                targs_ = collections.defaultdict(list)
                for t in targs:
                    targs_[t[0]] += [t[1]]
                targs_ = dict((k, np.max(v)) for k, v in targs_.items())
                targs = sorted(targs_.items(), key=lambda x: -x[1])
                dense = ",".join("%s(%d)" % x for x in targs)
                result = {
                    "inchikey": ik,
                    "raw": dense
                }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk
