import os

from chemicalchecker.util import logged

from .signaturizer import RawSignaturizerSetup
from .utils.tasks import Tasker
from .utils.log import set_logging
from .io import read_smiles_from_multiple_data


@logged
class MultiSetup(object):

    def __init__(self, 
                 data,
                 models_path,
                 **kwargs):
        self.models_path = os.path.abspath(models_path)
        os.makedirs(self.models_path, exist_ok=True)
        self.tasks = Tasker(data, models_path)

    def precalc_signatures(self, smiles_idx=1, sort=True, standardize=False, hpc=False, root=None, log="INFO", **kwargs):
        set_logging(log)
        if root is None:
            root = os.path.join(self.models_path, ".signatures")
            self.__log.info("Root set as %s" % root)
        os.makedirs(root, exist_ok=True)
        self.__log.info("Signaturizing from the multiple files")
        data_list = [t[0] for t in self.tasks]
        smiles = read_smiles_from_multiple_data(data_list, smiles_idx=smiles_idx, sort=sort, standardize=standardize)
        self.__log.info("Will signaturize %d molecules" % len(smiles.smiles))
        sign = RawSignaturizerSetup(root=root, hpc=hpc, **kwargs)
        print(sign)
        sign.signaturize(smiles.smiles)
        self.__log.info("Dictionary of signature paths done")
        sign_paths = sign.get_destination_dirs(**kwargs)
        return sign_paths

