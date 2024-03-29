import os

from chemicalchecker.util import logged

from .utils.tasks import Tasker
from .io import read_molecules_from_multiple_data

@logged
class MultiSetup(object):

    def __init__(self,
                 data,
                 models_path,
                 **kwargs):
        if type(models_path)== str:
            self.models_path = os.path.abspath(models_path)
            os.makedirs(self.models_path, exist_ok=True)
        else:
            models_path_=models_path
            self.models_path = []
            for m in models_path_:
                self.models_path += os.path.abspath(m)
                os.makedirs(m, exist_ok=True)
        self.tasks = Tasker(data, models_path)

    def precalc_signatures(self, molecule_idx=1, sort=True, standardize=False, hpc=False, root=None, log="INFO", inchi = False, **kwargs): # Pretty sure I'm not using this so leave as is for now
        # set_logging(log)
        if root is None:
            root = os.path.join(self.models_path, ".signatures")
            self.__log.info("Root set as %s" % root)
        os.makedirs(root, exist_ok=True)
        self.__log.info("Signaturizing from the multiple files")
        data_list = [t[0] for t in self.tasks]
        molecules = read_molecules_from_multiple_data(data_list, molecule_idx=molecule_idx, sort=sort, standardize=standardize, inchi = inchi)
        self.__log.info("Will signaturize %d molecules" % len(molecules.molecule))
        sign = RawSignaturizerSetup(root=root, hpc=hpc, **kwargs)
        sign.signaturize(smiles=molecules.molecules)
        self.__log.info("Dictionary of signature paths done")
        sign_paths = sign.get_destination_dirs(**kwargs)
        return sign_paths

