import os
import tempfile
from chemicalchecker.util import Config
from chemicalchecker.util.hpc import HPC
import multiprocessing


def cpu_count():
    return multiprocessing.cpu_count()


def fit_all_hpc(activity_path, models_path, **kwargs):
    """Run HPC jobs to fit all models for each activity file.

    Args:
        activity_path(str): Where TSV activity files are saved.
        models_path(str): Where models are saved.
        job_path(str): Path (usually in scratch) where the script files are
            generated.
    """
    # read config file
    cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
    cfg = Config(cc_config)
    # create job directory if not available
    job_base_path = cfg.PATH.CC_TMP
    tmp_dir = tempfile.mktemp(prefix='tmp_', dir=job_base_path)
    job_path = kwargs.get("job_path", tmp_dir)
    if not os.path.isdir(job_path):
        os.mkdir(job_path)
    # check cpus
    cpu = kwargs.get("cpu", 4)
    # create script file
    script_lines = [
        "import sys, os",
        "from tqdm import tqdm",
        "import pickle",
        "from chemicalchecker.tool.targetmate import TargetMate",
        "from chemicalchecker.core import ChemicalChecker",
        "cc = ChemicalChecker()",
        "task_id = sys.argv[1]",  # <TASK_ID>
        "filename = sys.argv[2]",  # <FILE>
        "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
        "s3_pred_fn = dict()",
        "for ds in cc.datasets:",
        "    s3 = cc.get_signature('sign3', 'full', ds)",
        "    s3_pred_fn[ds] = (s3, s3.get_predict_fn())",
        "for act_file in tqdm(inputs[task_id]):",
        "    act_file = str(act_file)",  # elements for current job
        "    act_name = os.path.splitext(os.path.basename(act_file))[0]",
        "    model_path = os.path.join('%s', act_name)" % models_path,
        "    if os.path.isfile(os.path.join(model_path,'ad_data.pkl')):",
        "        continue",
        "    tm = TargetMate(model_path, sign3_predict_fn=s3_pred_fn, n_jobs=%s)" % cpu,
        "    tm.fit(act_file)",
        "print('JOB DONE')"
    ]
    script_name = os.path.join(job_path, 'fit_all_targetmate.py')
    with open(script_name, 'w') as fh:
        for line in script_lines:
            fh.write(line + '\n')
    # hpc parameters
    elements = [os.path.join(activity_path, f)
                for f in os.listdir(activity_path)]
    params = {}
    params["num_jobs"] = kwargs.get("num_jobs", len(elements) / 100)
    params["jobdir"] = job_path
    params["job_name"] = "TARGETMATE_fit"
    params["elements"] = elements
    params["wait"] = False
    params["memory"] = 8
    params["cpu"] = cpu
    # job command
    singularity_image = cfg.PATH.SINGULARITY_IMAGE
    command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
        " OMP_NUM_THREADS={} singularity exec {} python {} <TASK_ID> <FILE>"
    command = command.format(
        os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config, str(cpu),
        singularity_image, script_name)
    # submit jobs
    cluster = HPC.from_config(Config())
    cluster.submitMultiJob(command, **params)
    return cluster


def predict_all_hpc(models_path, signature_path, results_path, **kwargs):
    """Run HPC jobs to predict with all models for input molecules.

    Args:
        models_path(str): Where models are saved.
        signature_path(str): Directory with all sign3 for molecules to
            predict.
        job_path(str): Path (usually in scratch) where the script files are
            generated.
        results_path(str): Path where to save predictions.
    """
    # read config file
    cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
    cfg = Config(cc_config)
    # create job directory if not available
    job_base_path = cfg.PATH.CC_TMP
    tmp_dir = tempfile.mktemp(prefix='tmp_', dir=job_base_path)
    job_path = kwargs.get("job_path", tmp_dir)
    if not os.path.isdir(job_path):
        os.mkdir(job_path)
    # check cpus
    cpu = kwargs.get("cpu", 4)
    # create script file
    script_lines = [
        "import sys, os",
        "from tqdm import tqdm",
        "import pickle",
        "from chemicalchecker.tool.targetmate import TargetMate",
        "task_id = sys.argv[1]",  # <TASK_ID>
        "filename = sys.argv[2]",  # <FILE>
        "inputs = pickle.load(open(filename, 'rb'))",  # load pickled data
        "data = None",
        "for mdl_dir in tqdm(inputs[task_id]):",
        "    mdl_dir = str(mdl_dir)",  # elements for current job
        "    mdl_name = os.path.normpath(mdl_dir).split('/')[-1]",
        "    result_file = os.path.join('%s', mdl_name)" % results_path,
        "    if os.path.isfile(result_file):",
        "        continue",
        "    if data is None:",  # trick to avoid re-parsing SMILES
        "        data = '%s'" % signature_path,
        "    tm = pickle.load(open(os.path.join(mdl_dir,'TargetMate.pkl'),'rb'))",
        "    tm.models_path = mdl_dir",
        "    mps, ad, prc, data = tm.predict(data=data,sign_folder='%s')" % signature_path,
        "    results = mps, ad, prc",
        "    pickle.dump(results, open(result_file,'wb'))",
        "print('JOB DONE')"
    ]
    script_name = os.path.join(job_path, 'predict_all_targetmate.py')
    with open(script_name, 'w') as fh:
        for line in script_lines:
            fh.write(line + '\n')
    # hpc parameters
    elements = [os.path.join(models_path, f)
                for f in sorted(os.listdir(models_path))]
    params = {}
    params["num_jobs"] = kwargs.get("num_jobs", len(elements) / 100)
    params["jobdir"] = job_path
    params["job_name"] = "TARGETMATE_predict"
    params["elements"] = elements
    params["wait"] = False
    params["memory"] = 8
    params["cpu"] = cpu
    # job command
    singularity_image = cfg.PATH.SINGULARITY_IMAGE
    command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" +\
        " OMP_NUM_THREADS={} singularity exec {} python {} <TASK_ID> <FILE>"
    command = command.format(
        os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config, str(cpu),
        singularity_image, script_name)
    # submit jobs
    cluster = HPC.from_config(Config())
    cluster.submitMultiJob(command, **params)
    return cluster
