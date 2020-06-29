import sys
import os
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py
from chemicalchecker import ChemicalChecker
from chemicalchecker.util.pipeline import Pipeline, PythonCallable, CCFit

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CC_CONFIG'] = os.path.join(current_dir,'configs/oncogenicity.json')
#os.environ['CC_ROOT'] = os.path.join('/aloy/home/lmateo/cc_oncogenicity/')

#CC_PATH = "/aloy/web_checker/package_cc/dream_ctd2/"
CC_PATH = "/aloy/home/lmateo/cc_oncogenicity/"

#pp = Pipeline(pipeline_path="/aloy/scratch/oguitart/dream_ctd2")
pp = Pipeline(pipeline_path="/aloy/scratch/sbnb-adm/cc_oncogenicity_pipeline")

# Input H5 (NS)
input_h5 = "/aloy/home/lmateo/cc_oncogenicity/sign0.h5"

# datasets = ['H1.001', 'H1.002', 'H1.003', 'H2.001',
#             'H2.003', 'H2.004', 'H2.005', 'H2.006', 'H2.007',
#             'H3.001', 'H3.002', 'H3.003']

size_target_keys = 32


def get_available_datasets(root="L", sign_type="sign3"):
    datasets = []
    path = "%s/full/%s" % (CC_PATH, root)
    datasets = []
    for f1 in os.listdir(path):
        path_1 = "%s/%s" % (path, f1)
        for f2 in os.listdir(path_1):
            path_2 = "%s/%s" % (path_1, f2)
            if os.path.exists("%s/%s/%s.h5" % (path_2, sign_type, sign_type)):
                datasets += [path_2.split("/")[-1]]
    return datasets


def projection_plot(dataset, sign_type="sign2"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cc = ChemicalChecker(CC_PATH)
    sign = cc.get_signature(sign_type, "full", dataset)
    sign1 = cc.get_signature('sign1', "full", dataset)
    if os.path.exists(sign1.data_path):
        keys_s2 = set(sign1.keys)
    else:
        sign0 = cc.get_signature('sign0', "full", dataset)
        keys_s2 = set(sign0.keys)
    keys_s = set(sign.keys)
    if len(keys_s) == len(keys_s2):
        keys, V = sign.get_vectors(sign.keys[-1000:])
    else:
        keys_1, V_1 = sign.get_vectors(sign.keys[-900:])
        intersect = list(keys_s2.intersection(keys_s))
        keys_2, V_2 = sign.get_vectors(intersect[:100])
        V = np.vstack([V_1, V_2])
        keys = np.concatenate([keys_1, keys_2])

    tsne = TSNE()
    P = tsne.fit_transform(V)
    mask = ["_" in k for k in keys]
    mask_s2 = [k in keys_s2 for k in keys]
    ax.scatter(P[:, 0], P[:, 1], s=10, alpha=0.5, color="gray")
    ax.scatter(P[mask_s2, 0], P[mask_s2, 1], color="blue", edgecolor="white")
    ax.scatter(P[mask, 0], P[mask, 1], color="red", edgecolor="white")

    ax.set_axis_off()
    ax.set_title(dataset + " (%s)" % sign_type)
    plt.tight_layout()
    plt.savefig(os.path.join(sign.stats_path, 'projections_plot.png'), dpi=300)


def intensity_plot(dataset, sign_type="sign2"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cc = ChemicalChecker(CC_PATH)
    sign = cc.get_signature(sign_type, "full", dataset)

    sign1 = cc.get_signature('sign1', "full", dataset)
    if os.path.exists(sign1.data_path):
        keys_s2 = set(sign1.keys)
    else:
        sign0 = cc.get_signature('sign0', "full", dataset)
        keys_s2 = set(sign0.keys)
    keys_s = set(sign.keys)
    if len(keys_s) == len(keys_s2):
        keys, V = sign.get_vectors(sign.keys[-1000:])
    else:
        keys_1, V_1 = sign.get_vectors(sign.keys[-900:])
        intersect = list(keys_s2.intersection(keys_s))
        keys_2, V_2 = sign.get_vectors(intersect[:100])
        V = np.vstack([V_2, V_1])
        keys = np.concatenate([keys_2, keys_1])

    #keys, V = sign.get_vectors(sign.keys[-1000:])
    mask_s2 = np.logical_and([k in keys_s2 for k in keys], [
                            "_" not in k for k in keys])
    I = np.sum(np.abs(V), axis=1)
    I_o = I[:-32]
    I_d = I[-32:]
    #print len(I[mask_s2]), len(I_o)
    ax.hist(I_o, density=True, zorder=100, color="gray", alpha=0.6)
    # ax.hist(I[mask_s2], density=True, zorder=100, color="blue", alpha=0.6)
    ax.hist(I_d, density=True, zorder=200, color="red", alpha=0.6)
    ax.set_title(dataset + " (%s)" % sign_type)
    ax.grid()
    ax.set_xlabel("Sum of signature values")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(sign.stats_path, 'intensity_plot.png'), dpi=300)


def auc_plots(dataset, sign_type="sign2"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cc = ChemicalChecker(CC_PATH)
    sign = cc.get_signature(sign_type, "full", dataset)
    from sklearn.metrics import auc
    path = sign.stats_path
    moa_file = os.path.join(path, "moa_%s_auc_validation.tsv" % sign_type)
    with open(moa_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        moa_fpr, moa_tpr = [], []
        for r in reader:
            moa_fpr += [float(r[0])]
            moa_tpr += [float(r[1])]
    atc_file = os.path.join(path, "atc_%s_auc_validation.tsv" % sign_type)
    with open(atc_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        atc_fpr, atc_tpr = [], []
        for r in reader:
            atc_fpr += [float(r[0])]
            atc_tpr += [float(r[1])]
    ax.plot(moa_fpr, moa_tpr, color="red", label="MoA (%.2f)" %
            auc(moa_fpr, moa_tpr), zorder=100)
    ax.plot(atc_fpr, atc_tpr, color="blue", label="ATC (%.2f)" %
            auc(atc_fpr, atc_tpr), zorder=50)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.legend()
    ax.grid()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(dataset + " (%s)" % sign_type)
    plt.savefig(os.path.join(sign.stats_path, 'moa_atc_plot.png'), dpi=300)


def make_eval_plots(datasets, sign_type):

    cc = ChemicalChecker(CC_PATH)
    cc_new = ChemicalChecker('/aloy/web_checker/package_cc/newpipe/')

    for ds in datasets:

        sign = cc.get_signature(sign_type, "full", ds)
        keys = sign.keys
        mask = [k for k in keys if '_' in k]

        if len(mask) != size_target_keys:
            print(len(mask))
            raise Exception('Missing target molecules for dataset ' + str(ds))

        if not os.path.exists(os.path.join(sign.stats_path, 'moa_' + sign_type + '_auc_validation.png')):

            sign.consistency_check()
            final_keys = []
            with h5py.File(sign.data_path, 'r') as hf:
                for i in range(hf['V'].shape[0]):
                    if np.sum(hf['V'][i]) == 0:
                        final_keys.append(hf['keys'][i])

            if len(final_keys) > 0:
                print("Number of keys empty are " + str(len(final_keys)))
            sign.validate()
            sign0_b4 = cc_new.get_signature(sign_type, "full", "B4.001")
            if os.path.exists(sign0_b4.data_path):
                sign.validate_versus_signature(sign0_b4)
        #intensity_plot(ds, sign_type)
        if not os.path.exists(os.path.join(sign.stats_path, 'projections_plot.png')):
            projection_plot(ds, sign_type)
            intensity_plot(ds, sign_type)
        if not os.path.exists(os.path.join(sign.stats_path, 'moa_atc_plot.png')):
            auc_plots(ds, sign_type)


datasets = get_available_datasets(sign_type='sign0')

#datasets.remove('H2.002')

s0_params = {}

s0_params["CC_ROOT"] = CC_PATH
s0_params['python_callable'] = make_eval_plots
s0_params['op_args'] = [datasets, 'sign0']

s0_task = PythonCallable(name="s0_val_plots", **s0_params)

pp.add_task(s0_task)

# NS: Add s0fit
s0f_params =  {'CC_ROOT': CC_PATH, 'data_file': input_h5}
s0f_task = CCFit(cc_type='sign0', **s0f_params)
pp.add_task(s0f_task)



s1_params = {}

s1_params["datasets"] = datasets
s1_params["CC_ROOT"] = CC_PATH
s1_params["full_reference"] = False
# s1_params["ds_params"] = {"H1.001": {"discrete": True},
#                           "H1.002": {"discrete": True},
#                           "H1.003": {"discrete": True},
#                           "H1.004": {"discrete": False},
#                           "H1.005": {"discrete": False},
#                           "H1.006": {"discrete": False},
#                           "H1.007": {"discrete": False},
#                           "H1.008": {"discrete": False},
#                           "H2.001": {"discrete": False},
#                           "H2.003": {"discrete": True},
#                           "H2.004": {"discrete": True},
#                           "H2.005": {"discrete": False},
#                           "H2.006": {"discrete": True},
#                           "H2.007": {"discrete": True},
#                           "H3.001": {"discrete": False},
#                           "H3.002": {"discrete": False},
#                           "H3.003": {"discrete": False},
#                           "H3.004": {"discrete": False},
#                           "H3.005": {"discrete": False},
#                           "H3.011": {"discrete": False},
#                           "H3.055": {"discrete": False}}

#s1_params["ds_params"] = {"L1.004": {"discrete": False}}

s1_task = CCFit(cc_type='sign1', **s1_params)

pp.add_task(s1_task)

s1_plots_params = {}

s1_plots_params["CC_ROOT"] = CC_PATH
s1_plots_params['python_callable'] = make_eval_plots
s1_plots_params['op_args'] = [datasets, 'sign1']

s1_plots_task = PythonCallable(name="s1_proj_plots", **s1_plots_params)

pp.add_task(s1_plots_task)

n1_params = {}

n1_params["CC_ROOT"] = CC_PATH
n1_params["datasets"] = datasets
n1_params["full_reference"] = False

n1_task = CCFit(cc_type='neig1', **n1_params)

pp.add_task(n1_task)

s2_params = {}
s2_params["CC_ROOT"] = CC_PATH
s2_params["datasets"] = datasets
s2_params["full_reference"] = False

s2_task = CCFit(cc_type='sign2', **s2_params)

pp.add_task(s2_task)

s2_plots_params = {}

s2_plots_params['python_callable'] = make_eval_plots
s2_plots_params['op_args'] = [datasets, 'sign2']

s2_plots_task = PythonCallable(name="s2_proj_plots", **s2_plots_params)

pp.add_task(s2_plots_task)

s3_params = {}

s3_params["CC_ROOT"] = CC_PATH
s3_params["target_datasets"] = datasets

s3_task = CCFit(cc_type='sign3', **s3_params)

pp.add_task(s3_task)

s3_plots_params = {}

s3_plots_params['python_callable'] = make_eval_plots
s3_plots_params['op_args'] = [datasets, 'sign3']

s3_plots_task = PythonCallable(name="s3_proj_plots", **s3_plots_params)

pp.add_task(s3_plots_task)

s4_plots_params = {}

datasets = get_available_datasets(sign_type='sign4')

# datasets.remove('H2.002')
s4_plots_params["CC_ROOT"] = CC_PATH
s4_plots_params['python_callable'] = make_eval_plots
s4_plots_params['op_args'] = [datasets, 'sign4']

s4_plots_task = PythonCallable(name="s4_proj_plots", **s4_plots_params)

pp.add_task(s4_plots_task)
n1_task.clean()
s1_task.clean()
s2_task.clean()
s0_task.clean()
s3_task.clean()
s1_plots_task.clean()
s2_plots_task.clean()
s3_plots_task.clean()
s4_plots_task.clean()
pp.run()
