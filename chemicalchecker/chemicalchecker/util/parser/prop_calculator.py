import os


from chemicalchecker.util import logged
import timeout_decorator
try:
    from e3fp import pipeline
    from rdkit.Chem import AllChem as Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import MACCSkeys
except ImportError:
    pass


@logged
class PropCalculator():
    """Container for static parsing methods.

    A parsing function here is iterating on an input file. It has to define
    on each input line the source id and the smile of  a molecule. Then the
    smile is converted to inchi and inchikey. The lines are appended as
    dictionaies and yielded in chunks.
    """

    @staticmethod
    def calc_fn(function):
        try:
            return eval('PropCalculator.' + function)
        except Exception as ex:
            PropCalculator.__log.error(
                "Cannot find calculator function %s", function)
            raise ex

    @staticmethod
    def fp2d(inchikey_inchi, chunks=1000):
        iks = inchikey_inchi.keys()
        nBits = 2048
        radius = 2
        chunk = list()
        for ik in iks:
            v = inchikey_inchi[ik]
            # PropCalculator.__log.info( ik)
            mol = Chem.rdinchi.InchiToMol(v)[0]
            info = {}
            # print mol
            fp = Chem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=nBits, bitInfo=info)
            dense = ",".join("%d" % s for s in sorted(
                [x for x in fp.GetOnBits()]))
            result = {
                "inchikey": ik,
                "raw": dense
            }
            chunk.append(result)
            if len(chunk) == chunks:
                yield chunk
                chunk = list()
        yield chunk

    @staticmethod
    def fp3d(inchikey_inchi, chunks=1000):

        params = pipeline.params_to_dicts(root + "/../files/defaults.cfg")

        @timeout_decorator.timeout(100, use_signals=False)
        def fprints_from_inchi(inchi, inchikey, confgen_params={}, fprint_params={}, save=False):
            mol = Chem.rdinchi.InchiToMol(inchi)[0]

            if Descriptors.MolWt(mol) > 800 or rdMolDescriptors.CalcNumRotatableBonds(mol) > 11:
                return None

            smiles = Chem.MolToSmiles(mol)
            return pipeline.fprints_from_smiles(smiles, inchikey, confgen_params, fprint_params, save)

        iks = inchikey_inchi.keys()
        chunk = list()
        for k in iks:
            try:
                fps = fprints_from_inchi(
                    inchikey_inchi[k], k, params[0], params[1])
            except Exception:
                PropCalculator.__log.warning('Timeout inchikey: ' + k)
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
    def scaffolds(inchikey_inchi, chunks=1000):

        iks = inchikey_inchi.keys()
        nBits = 1024
        radius = 2

        def murcko_scaffold(mol):
            core = MurckoScaffold.GetScaffoldForMol(mol)
            if not Chem.MolToSmiles(core):
                core = mol
            fw = MurckoScaffold.MakeScaffoldGeneric(core)
            info = {}
            c_fp = Chem.GetMorganFingerprintAsBitVect(
                core, radius, nBits=nBits, bitInfo=info).GetOnBits()
            f_fp = Chem.GetMorganFingerprintAsBitVect(
                fw, radius, nBits=nBits, bitInfo=info).GetOnBits()
            fp = ["c%d" % x for x in c_fp] + ["f%d" % y for y in f_fp]
            return ",".join(fp)

        chunk = list()
        for k in iks:
            v = inchikey_inchi[k]
            mol = Chem.rdinchi.InchiToMol(v)[0]
            try:
                dense = murcko_scaffold(mol)
            except:
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
    def subskeys(inchikey_inchi, chunks=1000):

        iks = inchikey_inchi.keys()
        chunk = list()
        for k in iks:
            v = inchikey_inchi[k]
            mol = Chem.rdinchi.InchiToMol(v)[0]
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
