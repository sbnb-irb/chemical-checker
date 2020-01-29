import tempfile
import os
import shutil

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseTask
from chemicalchecker.util import HPC


@logged
class ExemplaryLinks(BaseTask):

    def __init__(self, config, name, **params):

        BaseTask.__init__(self, config, name, **params)

    def run(self):
        """Run the molprops step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)

        dataset_codes = list()
        for ds in all_datasets:
            if not ds.exemplary:
                continue

            dataset_codes.append(str(ds.dataset_code))

        target_path = os.path.join(config_cc.PATH.CC_ROOT, "exemplary")

        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        for code in dataset_codes:

            signature_path = cc.get_signature_path("sign1", "full", code)

            source_path = signature_path[:-6]

            target_dir = os.path.join(target_path, code[:1])

            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

            if not os.path.exists(os.path.join(target_dir, code[:2])):
                os.symlink(source_path, os.path.join(target_dir, code[:2]))

        self.mark_ready()
