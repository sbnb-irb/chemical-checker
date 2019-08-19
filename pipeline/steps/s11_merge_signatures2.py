import tempfile
import os
import shutil
import collections

from chemicalchecker.util import logged
from chemicalchecker.database import Dataset
from chemicalchecker.util import Config
from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util import BaseStep
from chemicalchecker.util import HPC


@logged
class MergeSignatures2(BaseStep):

    def __init__(self, config, name, **params):

        BaseStep.__init__(self, config, name, **params)

    def run(self):
        """Run the merge sign2 step."""

        all_datasets = Dataset.get()
        config_cc = Config()

        cc = ChemicalChecker(config_cc.PATH.CC_ROOT)
        map_coords = collections.defaultdict(list)
        new_datasets_ae = dict()
        for ds in all_datasets:
            if not ds.essential:
                continue

            map_coords[ds.coordinate] += [ds.dataset_code]

        for coord, datasets in map_coords.items():

            new_ds = coord + ".000"

            print new_ds

            sign2 = cc.get_signature("sign2", "full", new_ds)
            if sign2.is_fit():
                continue

            if os.path.exists(sign2.signature_path):
                print sign2.signature_path
                shutil.rmtree(sign2.signature_path)
                shutil.rmtree(sign2.signature_path[:-5])

            if len(datasets) > 1:
                new_datasets_ae[new_ds] = datasets
            else:
                sign2_old = cc.get_signature("sign2", "full", datasets[0])
                os.symlink(sign2_old.signature_path[
                           :-6], sign2.signature_path[:-6])

        job_path = tempfile.mkdtemp(
            prefix='jobs_AE_sign2_', dir=self.tmpdir)

        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # create script file
        cc_config_path = os.environ['CC_CONFIG']
        cc_package = os.path.join(config_cc.PATH.CC_REPO, 'package')
        script_lines = [
            "import sys, os",
            "import pickle",
            "import h5py",
            "import numpy as np",
            # cc_config location
            "os.environ['CC_CONFIG'] = '%s'" % cc_config_path,
            "sys.path.append('%s')" % cc_package,  # allow package import
            "from chemicalchecker.util import Config",
            "from chemicalchecker.core import ChemicalChecker",
            "from chemicalchecker.tool.autoencoder import AutoEncoder",
            "config = Config()",
            "task_id = sys.argv[1]",  # <TASK_ID>
            "filename = sys.argv[2]",  # <FILE>
            "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
            "new_ds_merge = inputs[task_id]",  # elements for current job
            # elements are indexes
            'cc = ChemicalChecker(config.PATH.CC_ROOT )',
            'map_ds = dict()',
            'keys = set()',
            'list_keys = list()',
            'new_ds, list_ds = new_ds_merge.popitem()',
            'for ds in list_ds:',
            '    map_ds[ds] = cc.get_signature("sign2","full",ds)',
            '    keys.update(map_ds[ds].keys)',
            'for key in keys:',
            '    list_keys.append(str(key))',
            'list_keys.sort()',
            'sign2_new = cc.get_signature("sign2", "full", new_ds)',
            'output_file = os.path.join(sign2_new.signature_path, "map_full.h5")',
            'with h5py.File(output_file, "w") as hf:',
            '    hf.create_dataset("keys", data=np.array(list_keys))',
            '    hf.create_dataset("x", (len(list_keys), 128*len(list_ds)), dtype=np.float32)',
            'chunk_size = 1000',
            'map_signs = dict()',
            'for i in range(0, len(list_keys), chunk_size):',
            '    chunk = slice(i, i + chunk_size)',
            '    keys = list_keys[chunk]',
            '    for b in list_ds:',
            '        _, map_signs[b] = map_ds[b].get_vectors(keys, include_nan=True)',
            '    rows = list()',
            '    for j in range(len(keys)):',
            '        rows.append(np.hstack([map_signs[b][j] for b in list_ds]))',
            "    with h5py.File(output_file, 'r+') as hf:",
            '        hf["x"][chunk] =  np.vstack(rows)',
            'params = {"cpu": 32, "epochs": 1000, "encoding_dim": 128, "mask_value" : np.nan}',
            'ae = AutoEncoder(sign2_new.model_path, **params)',
            'ae.fit(output_file)',
            'ae.encode(output_file,sign2_new.data_path, input_dataset="x")',
            "sign2_new.validate()",
            "sign2_new.mark_ready()",
            "print('JOB DONE')"
        ]

        script_name = os.path.join(job_path, 'sign2_ae_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line + '\n')
        # hpc parameters

        params = {}
        params["num_jobs"] = len(new_datasets_ae)
        params["jobdir"] = job_path
        params["job_name"] = "CC_AE_S2"
        params["elements"] = new_datasets_ae
        params["wait"] = True
        params["memory"] = 20
        params["cpu"] = 32
        # job command
        singularity_image = Config().PATH.SINGULARITY_IMAGE
        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(config_cc)
        jobs = cluster.submitMultiJob(command, **params)

        dataset_not_done = []

        for coord, datasets in map_coords.items():

            new_ds = coord + ".000"

            sign2 = cc.get_signature("sign2", "full", new_ds)
            if sign2.is_fit():
                continue

            dataset_not_done.append(new_ds)
            self.__log.warning(
                "Signature2 autoencoder failed for dataset code: " + new_ds)

        if len(dataset_not_done) == 0:
            self.mark_ready()
            shutil.rmtree(job_path)
