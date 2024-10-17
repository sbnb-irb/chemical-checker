
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
import os
import h5py
import tempfile
import pickle

from chemicalchecker.core.signature_base import BaseSignature
from chemicalchecker.core.signature_data import DataSignature
from chemicalchecker.util import Config, logged
from chemicalchecker.util.hpc import HPC
from chemicalchecker.util.parser import Converter
from chemicalchecker.util.plot.util import *
#from signaturizer import Signaturizer

from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score, precision_score, hamming_loss, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors


import logging
from tqdm import tqdm

@logged
class char(BaseSignature, DataSignature):
    """Enrichment signature class"""
    
    def __init__(self, signature_path, dataset, **params):
        """Initialize a visualization class.

        Args:
            signature_path(str): The path to the signature directory.
            dataset(object): The dataset object with all info related.
            metric(str): The metric used for the SAFE algorithm: euclidean or cosine (default: cosine).
        """
                
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **params)
        
        # Get the mapping between features and descriptions
        self.get_dict()
        
        # Get the metric (euclidean or cosine) from kwargs
        self.metric = params.pop('metric', 'cosine')     

        for param, value in params.items():
            self.__log.debug('parameter %s : %s', param, value)
            
        # Define all the directories and files where the data will be stored       
        self.data_path   = os.path.join(signature_path, f"visu_{self.metric}.h5")
        self.model_path  = os.path.join(signature_path, self.metric, "models")
        self.diags_path  = os.path.join(signature_path, self.metric, 'diags')
        self.stats_path  = os.path.join(signature_path, self.metric, 'stats')
        self.safe_path   = os.path.join(self.model_path, "safe.h5")
        self.scores_path = os.path.join(self.model_path, 'scores.h5')
        self.proj_path   = os.path.join(self.signature_path, self.metric, 
                                        f'projection{self.cctype[-1]}')

        self.__log.debug('signature path is: %s', signature_path)
        
        # Create the directories
        os.makedirs(signature_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.diags_path, exist_ok=True)
        os.makedirs(self.stats_path, exist_ok=True)

        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)
        
    def get_dict(self):
        """Get mappings between feature tags and their descriptions."""
        config_path = os.environ['CC_CONFIG']
        
        cc_repo = os.path.abspath(Config(config_path).PATH.CC_REPO)
        
        dict_path = os.path.join(cc_repo, 'package/chemicalchecker/tool', 
                                 'cc_charts/feature_description_mappings',
                                 self.dataset[:2])        
        try:
            with open(dict_path, 'rb') as fh:
                self.space_dict = pickle.load(fh)
        
        except FileNotFoundError as e:
            self.__log.warning(f'Could not find the feature-description mapping file. Continuing without the mapping.')
            sign0 = self.get_sign(
                f'sign0').get_molset(self.molset)
            self.space_dict = {f: f for f in sign0.features}

    def get_h5_attr(self, h5_dataset_name):
        """Get a specific attribute in the signature."""
        self.__log.debug("Fetching attribute %s" % h5_dataset_name)
        self._check_data()
        with h5py.File(self.data_path, 'r') as hf:
            if h5_dataset_name not in hf.attrs.keys():
                raise Exception("HDF5 file has no '%s'." % h5_dataset_name)
            if hasattr(hf.attrs[h5_dataset_name], 'decode'):
                data = hf.attrs[h5_dataset_name].astype(str)
            else:
                data = hf.attrs[h5_dataset_name]

        return data
    
    def add_attr(self, data_dict, overwrite=True):
        """Add dataset to a H5"""
        for k, v in data_dict.items():
            with h5py.File(self.data_path, 'a') as hf:
                if k in hf.attrs.keys():
                    if overwrite:
                        del hf.attrs[k]
                    else:
                        self.__log.info('Skipping `%s~`: already there')
                        continue
                if isinstance(v, list):
                    if hasattr(v[0], 'decode') or isinstance(v[0], str) or isinstance(v[0], np.str_):
                        v = self.h5_str(v)
                else:
                    if hasattr(v, 'decode') or isinstance(v, str) or isinstance(v, np.str_):
                        v = self.h5_str(v)
                hf.attrs.create(k, data=v)        

    def fit(self, safe, sign0=None, sign1=None, back_dist_pvalue=0.01):
        """Fit the visualization class. A SAFE analysis is performed over the signature 4 of those
        molecules with available signature 0. This is followed by a tSNE of the resulting scores. Finally,
        the projected molecules are clustered by HDBSCAN.
        
        Args:
            safe(bool): A boolean indicating whether to perform the SAFE analysis or not. This is useful in case the
            instance has already been fitted and we want to change the downstream analysis without repeating the SAFE results.
            sign0(object): Signature 0 of the dataset of interest.
            sign1(object): Signature 1 of the dataset of interest.
            back_dist_pvalue(float): Distance p-value threshold for a molecule to be considered as close when searching for 
            neighbors in the SAFE analysis (default: 0.01)."""
        from hdbscan import HDBSCAN, approximate_predict
        
        BaseSignature.fit(self, overwrite=True)
        
        # Load all the necessary signatures
        self.__log.info("Loading the data.")
        if sign0 is None:
            sign0 = self.get_sign(
                f'sign0').get_molset(self.molset)
        
        if sign1 is None:
            sign1 = self.get_sign(
                f'sign{self.cctype[-1]}').get_molset(self.molset)
             
        # Check the availability of the signatures        
        if os.path.isfile(sign0.data_path):
            features = sign0.features.astype(str)
        
        else:
            raise Exception("The file " + sign0.data_path + " does not exist")

        if os.path.isfile(sign1.data_path):
             pass
            
        else:
            raise Exception("The file " + sign1.data_path + " does not exist")
            
        tmp_dir = tempfile.mkdtemp(
            prefix='visu_' + self.dataset + "_", dir=Config().PATH.CC_TMP)
        
        self.__log.debug("Temporary files saved in " + tmp_dir)
        
                
        # Get the signatures of those molecules with available signature 0       
        self.__log.info(f'Intersecting the experimental data and signature {self.cctype[-1]}.')
        
        keys, V, V0 = sign1.get_intersection(sign0)
        
        # Binarize the experimental data (in case it is ternary)
        V0[V0 > 1] = 1
                
        # Save signature data
        self.add_datasets(dict(keys=keys,
                               V0=V0.astype(np.bool), 
                               V=V, 
                               features=features)
                         )
        
        if safe:  
            
            # Find neighborhoods
            self.__log.info("Finding local neighborhoods.")
            self.func_hpc('get_neighborhoods', back_dist_pvalue, cpu=32, mem_by_core=5, wait=True)
            
            # Run SAFE
            self.__log.info('Running SAFE')
            self.run_SAFE()
        
        self.thr = self.get_h5_attr('thr')
        scores = self.get_h5_dataset('scores')
        
        # Project the signatures
        self.__log.info('Computing the tSNE projection of the signatures')
        self.func_hpc('project_scores', cpu=16, mem_by_core=5, wait=True)
        
        safe_coords = self.get_h5_dataset('safe_coords')
        
        self.plot_neighborhoods()

        self.__log.error('Plotting the projection')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        palette = make_cmap([(1, 1, 1), (0.2, 0.2, 0.2)])
        sns.kdeplot(safe_coords[:, 0], 
                    safe_coords[:,1],
                    levels=100, 
                    c='grey', 
                    cmap=palette,
                    alpha=0.6,
                    fill=True, 
                    ax=ax)
        
        with open(os.path.join(self.diags_path, "safe_proj.pkl"), 'wb') as fh:
            pickle.dump(fig, fh)
            
        self.__log.info('Cluster analysis')
        
        self.cluster_analysis()
            
        self.mark_ready()
        
    def get_neighborhoods(self, back_dist_pvalue):
        
        V1 = self.get_h5_dataset('V')
        
        # Get distance corresponding to the specified p-value
        sign1 = self.get_sign(
                f'sign{self.cctype[-1]}').get_molset(self.molset)
        back_dict = sign1.background_distances(self.metric)
        radius = back_dict['distance'][back_dict['pvalue']==back_dist_pvalue]            
        
        # Find all neighbors within the distance threshold
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(radius=radius, metric=self.metric, n_jobs=-1)
        nn.fit(V1)
        dist, nn_idxs = nn.radius_neighbors(V1)
        
        with h5py.File(self.safe_path, 'w') as f:
            
            # Remove the molecules themselves from their nearest neighbors
            for i, nearest in enumerate(nn_idxs):
                nn_idxs[i] = np.delete(nearest, (dist[i]==0).nonzero()[0])
                dist[i] = np.delete(dist[i], (dist[i]==0).nonzero()[0])
                
                # Store nearest neighbors indexes as h5 subgroups
                f.create_dataset(f'neighbors/{i}', data=nearest, dtype=int)

        neigh_lengths = np.array(list(map(len, nn_idxs)))
        n_isolated = len(neigh_lengths[neigh_lengths==0])
        coverage = n_isolated/neigh_lengths.shape[0]
        
        self.add_attr(dict(radius=radius, neigh_coverage=coverage))
           
    def run_SAFE(self, elements=None):
        """Parallelizes the enrichment analysis making use of the HPC.
        
        Args:
            elements(list): A list containing the column indexes of the features of 
            the experimental data that we want to analyze. Only useful for re-running
            failed jobs. By default, all the features are analyzed."""
        
        # Create the folder to store the SAFE results       
        res_folder = os.path.join(self.model_path, 'safe')
        os.makedirs(res_folder, exist_ok=True)
        
        # Get the shape of the results matrix (same as the shape of the experimental data)
        with h5py.File(self.data_path) as f:
            m = f['V0'].shape[0]   # Number of rows
            n = f['V0'].shape[1]   # Number of columns
        
        if elements is None:
            elements = [[self.safe_path, col_idx, self.data_path, res_folder] for col_idx in range(n)]
        
        tmp_dir = tempfile.mkdtemp(
            prefix='visu_' + self.dataset + "_", dir=Config().PATH.CC_TMP)
        self.__log.debug(f"temporary files of the SAFE analysis are stored in {tmp_dir}")
        
        # HPC parameters
        params = {}
        params['job_name'] = 'enrichment'
        params["jobdir"] = tmp_dir
        params['cpu'] = 1
        params["wait"] = True
        params["elements"] = elements
        params["num_jobs"] = len(elements)
        params["max_jobs"] = 120
        
        cc_config = os.environ['CC_CONFIG']
        cfg = Config(cc_config)
        
        script_name = os.path.join(cfg.PATH.CC_REPO, 'package', 'chemicalchecker', 'core', 'char_utils', 'fisher.py')

        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" \
            " singularity exec {} python {} <TASK_ID> <FILE>"
        
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)

        # Submit jobs to the HPC
        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        
        # Check for failed jobs and re-run them.
        while True:
            missing_cols = list()     
            for col_idx in range(n):
                if not os.path.exists(os.path.join(res_folder, str(col_idx))):
                    missing_cols.append(col_idx)

            if missing_cols:
                self.__log.error(f'Re-running {len(missing_cols)} jobs.')
                elements = [[self.safe_path, col_idx, self.data_path, res_folder] for col_idx in missing_cols]
                self.run_SAFE(elements)
                return

            else:
                break

        # Read pickled results and store them in an h5
        self.__log.info("Saving SAFE results.")
        with h5py.File(self.data_path, 'a') as f:
          
            raw_scores = np.zeros(shape=(n, m), dtype='float32')
            for col_idx in range(n):
                with open(os.path.join(res_folder, str(col_idx)), 'rb') as fh:
                    data_row = pickle.load(fh)
                    raw_scores[col_idx] = data_row
            
            raw_scores = raw_scores.T
            
            # Impute infinite scores as the maximum non-infinite score
            max_score = raw_scores[raw_scores!=np.inf].max()
            self.add_attr(dict(max_score=max_score))
            raw_scores[np.isinf(raw_scores)] = max_score
            self.add_datasets(dict(raw_scores=raw_scores))
            scores = raw_scores/max_score  # Transform to a [0,1] range
            self.add_datasets(dict(raw_scores=raw_scores, 
                                   scores=scores))

            # Compute the optimal threshold for the scores (the one recapitulating 
            # better the experimental data in terms of the F1-score).
            thr = self.find_thr()
            
            # Get positive features according to the optimal threshold
            enriched = (scores >= thr)
            self.add_datasets(dict(enriched=enriched))
            
        # Remove pickles
        import shutil
        shutil.rmtree(res_folder)
        
    def find_thr(self, stat='fscore', n_max_samples=10000):
        """Find the score threshold that better recapitulates the signature 0 data."""

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_recall_curve, roc_curve, auc, plot_precision_recall_curve
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
       
        keys = self.keys
        max_score = self.get_h5_attr('max_score')
        
        # Take a subsample if the dataset is too big
        if n_max_samples < len(keys):
            self.__log.warning('Subsampling 10000 molecules to compute the ROC and PR curves')
            idxs = sorted(np.random.choice(
                len(keys), n_samples, replace=False))
            y_true = self.get_h5_dataset('V0', mask=np.array(idxs))
            y_true = self.get_h5_dataset('raw_scores', mask=np.array(idxs))

        else:
            y_true = self.get_h5_dataset('V0')
            y_prob = self.get_h5_dataset('raw_scores')

        y_prob = -np.log10(y_prob)
        y_prob[np.isinf(y_prob)] = max_score

        y_true_f = y_true.flatten()
        y_prob_f = y_prob.flatten()

        # Compute the evaluation curves
        precision, recall, pr_thr = precision_recall_curve(y_true_f, y_prob_f)
        fpr, tpr, roc_thr = roc_curve(y_true_f, y_prob_f)

        if stat=='gmean':
            gmeans = np.sqrt(tpr * (1-fpr))            
            thresholds = roc_thr
            ix = np.nanargmax(gmeans)

        else:
            fscore = (2 * precision * recall) / (precision + recall)
            thresholds = pr_thr
            ix = np.nanargmax(fscore)
        
        thr = thresholds[ix]/max_score
        self.add_attr(dict(thr=thr))
        
        # Iterpolate and accumulate to smooth the curve
        interp_x = np.linspace(0, 1, 2000)
        tpr = np.interp(interp_x, fpr, tpr)
        decreasing_max_precision = np.maximum.accumulate(precision)
        precision = np.interp(interp_x, 
                              recall[::-1], 
                              decreasing_max_precision[::-1])

        auroc = auc(interp_x, tpr)
        aupr = auc(interp_x, precision)

        # Expected precision
        expected_pr = len(y_true_f[y_true_f==1]) / len(y_true_f)

        color = cc_colors(self.dataset[:2])
        
        y_pred = (y_prob >= thr)
        y_pred_f = y_pred.flatten()

        conf = confusion_matrix(y_true_f, y_pred_f)
        TN = conf[0, 0]
        FN = conf[1, 0]
        TP = conf[1, 1]
        FP = conf[0, 1]

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        PRECISION = TP/(TP+FP)


        fig, ax = plt.subplots(figsize=(5,5))


        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
        ax.plot(interp_x, tpr, color='grey', label=f'AUROC = {auroc:.2f}')
        ax.fill_between(interp_x, tpr, color=color, alpha=0.25)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(fontsize=14)
        ax.set_title(self.dataset[:2], fontsize=16)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.scatter(FPR, TPR, marker='o', color='red', label=f'Thr = {thr:.2f}', zorder=999)
        ax.legend(fontsize=14)

        fig.savefig(os.path.join(self.diags_path, 'ROC_curve.png'), 
                    bbox_inches='tight',
                    dpi=300)

        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot(interp_x, precision, color='grey', label=f'AUPR = {aupr:.2f}')
        ax.fill_between(interp_x, precision, color=color, alpha=0.25)
        ax.scatter(TPR, PRECISION, marker='o', color='red', label=f'Optimal thr = {thr:.2f}', zorder=999)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=14)
        ax.set_title(self.dataset[:2], fontsize=16)
        ax.set_aspect('equal')
        ax.grid(True)

        fig.savefig(os.path.join(self.diags_path, 'PR_curve.png'), 
                    bbox_inches='tight',
                    dpi=300)

        accuracy = accuracy_score(y_true_f, y_pred_f)
        micro_precision = precision_score(y_true, y_pred, average='micro')
        sample_precision = precision_score(y_true, y_pred, average='samples')
        micro_recall = recall_score(y_true, y_pred, average='micro')
        sample_recall = recall_score(y_true, y_pred, average='samples')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        sample_f1 = f1_score(y_true, y_pred, average='samples')        

        fig, ax = plt.subplots(figsize=(5,5))

        ax.bar(np.linspace(0, 1, 7), 
                  [accuracy, 
                   micro_precision, 
                   sample_precision, 
                   micro_recall, 
                   sample_recall, 
                   micro_f1, 
                   sample_f1
                  ],
                  width=float(1/7),
                  color=color, alpha=0.6)
        ax.set_xticks(np.linspace(0, 1, 7))
        ax.set_xticklabels(['Accuracy', 
                            'Sample precision', 
                            'Micro-av. recision', 
                            'Sample recall', 
                            'Micro-av. recall', 
                            'Sample F1', 
                            'Micro-av. F1'], 
                           rotation=60, 
                           ha='right')

        ax.set_aspect('equal')
        ax.set_ylim(0, 1)
        ax.grid(True)

        fig.savefig(os.path.join(self.diags_path, 'safe_metrics.png'),
                    bbox_inches='tight',
                    dpi=300)        

        with open(os.path.join(self.stats_path, 'safe_metrics'), 'wb') as fh:
            metrics = dict()
            metrics['pr_thr'] = thresholds[ix]/max_score
            metrics['precision'] = precision
            metrics['recall'] = interp_x
            metrics['f1'] = fscore
            metrics['tpr'] = tpr
            metrics['auroc'] = auroc
            metrics['aupr'] = aupr
            metrics['pr_opt_coords'] = (TPR, PRECISION)
            metrics['roc_opt_coords'] = (FPR, TPR)
            metrics['expected_pr'] = expected_pr
            pickle.dump(metrics, fh)


        return thresholds[ix]/max_score

    def project_scores(self):
        """Use the CC projection's module to compute the tSNE coordinates of the enrichment scores."""
        from chemicalchecker.core import DataSignature
        from chemicalchecker.core.proj import proj
        from sklearn.preprocessing import StandardScaler
        scores_path = os.path.join(self.model_path, 'scores.h5')
        
        scores = self.get_h5_dataset('scores')
        V = self.get_h5_dataset('V')
        
        with h5py.File(scores_path, 'w') as f:
            f.create_dataset('V', data=scores)
            f.create_dataset('keys', data=np.array(self.keys, 
                                                       DataSignature.string_dtype())) 
            
        if os.path.isfile(os.path.join(self.proj_path, 'models', 'Default', 'fit.ready')):
            os.remove(os.path.join(self.proj_path, 'models', 'Default', 'fit.ready'))
        
        s3 = DataSignature(self.scores_path)
        s3.molset = self.molset
        s3.dataset = self.dataset
        s3.cctype = self.cctype

        projector = proj(self.proj_path, s3, cpu=16)
        
        projector.fit(s3, validations=False, preprocess_dims=False)
        
        safe_coords = projector.data
        
        self.add_datasets(dict(safe_coords=safe_coords))
                
        return projector, safe_coords      
        
    def plot_projection(self):
        """Plot and store a projection of the whole space (used as a background for the visualizations)."""
        
        safe_coords = self.get_h5_dataset('safe_coords')
        fig, ax = plt.subplots(figsize=(10, 10))
        
        palette = make_cmap([(1, 1, 1), (0.2, 0.2, 0.2)])
        sns.kdeplot(safe_coords[:, 0], 
                    safe_coords[:,1],
                    levels=100, 
                    c='grey', 
                    cmap=palette,
                    alpha=0.6,
                    fill=True, 
                    ax=ax)
        
        with open(os.path.join(self.diags_path, "safe_proj.pkl"), 'wb') as fh:
            pickle.dump(fig, fh)
                    
    def plot_neighborhoods(self, s=10):
        
        with h5py.File(self.data_path) as f:
            m = f['V'].shape[0]   # Number of rows
            n = f['V0'].shape[1]  # Number of columns
        
        nn_idxs = list()
        
        if m > 10000:
            random_idxs = sorted(np.random.choice(
                len(self.keys), 1000, replace=False))
           
        else:
            random_idxs = range(m)
        
        with h5py.File(self.safe_path) as f:
            for n in tqdm(random_idxs):
                nn_idxs.append(f[f'neighbors/{n}'][:])
        
        fig, ax = plt.subplots(figsize=(5, 5))
        neigh_lengths = np.array(list(map(len, nn_idxs)))

        from collections import Counter
        occurence_count = Counter(neigh_lengths)
        
        sns.kdeplot(neigh_lengths, color=cc_colors(self.dataset[:2]), alpha=0.25, cut=0, fill=True, ax=ax)
        
        with h5py.File(self.data_path, 'a') as f:
            f.require_dataset('neighborhood_sizes', shape=neigh_lengths.shape, dtype=int,
                              data=neigh_lengths)

        ax.set_xlabel('Number of neighbors')
        ax.set_ylabel('')
        ax.set_title(self.dataset[:2], fontsize=16)
        from matplotlib.ticker import FormatStrFormatter

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.set_title('Number of neighbors per neighborhood')

        fig.savefig(os.path.join(self.diags_path, 'neighs_kde.png'), dpi=300)
        
        coords = self.get_h5_dataset('safe_coords')[random_idxs]
        
        fig, ax = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={'height_ratios': (1, 20), 'hspace': 0.1})
        max_n = round(np.percentile(neigh_lengths, 75), ndigits=-1)
        if max_n < 6:
            max_n = 6
        neigh_lengths[neigh_lengths >= max_n] = max_n
        projection(coords, front_kwargs=[dict(c=neigh_lengths, s=s, edgecolors='none', cmap='viridis')], ax=ax[1])
        ax[1].set_aspect('equal')
        from matplotlib import colorbar
        cbar = colorbar.ColorbarBase(ax[0], orientation='horizontal',
                                         ticklocation='top', cmap=cm.viridis)
        cbar.ax.set_xlabel('Number of neighbors', labelpad=10, rotation=0)
        cbar.ax.tick_params(axis='x', pad=0)
        cbar.set_ticks([1, .8, .6, .4, .2, .0])
        cbar.set_ticklabels([f'>= {max_n}'] + [f'{x:.0f}' for x in np.linspace(neigh_lengths.max(), neigh_lengths.min(), 6)][1:])

        cbar.ax.set_aspect(0.05)

        fig.savefig(os.path.join(self.diags_path, 'neigh_lengths_tsne.png'), dpi=300)

    def cluster_analysis(self, min_cluster_size=25):

        import mpld3
        from hdbscan import HDBSCAN, all_points_membership_vectors
        
        scores = self.get_h5_dataset('scores')
        enriched = self.get_h5_dataset('enriched')
        safe_coords = self.get_h5_dataset('safe_coords')
        thr = self.get_h5_attr('thr')
                    
        
        # Get the median of the neighborhood sizes as the
        # minimum cluster size
        neigh_sizes = self.get_h5_dataset('neighborhood_sizes')
        min_cluster_size = int(np.median(neigh_sizes))

        def cluster_enrichment(min_cluster_size=min_cluster_size):
            self.__log.info(f'Min. cluster size: {min_cluster_size}')
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                                core_dist_n_jobs=-1, 
                                prediction_data=True)
            clusters = clusterer.fit_predict(safe_coords)
            
            return clusterer, clusters

        def compute_scores(clusterer, clusters, get_metrics=True):
            
            enriched = self.get_h5_dataset('enriched')
            
            # Get vectors of cluster membership probabilities
            memb = all_points_membership_vectors(clusterer)
            
            strings = list()
            centroids = list()
            coverages = list()
            cluster_feats = list()
            cluster_labels = dict()
            clusters_of_interest = list()
            for cluster in np.unique(clusters):
                
                # Get features enriched for the molecules in the same cluster
                cluster_enriched = enriched[clusters==cluster]
                feature_count = cluster_enriched.sum(0)
                mask = feature_count.astype(bool)
                cluster_features = self.features[mask]
                feature_count = feature_count[mask]
                
                # Filter features that cover less than 75% of the cluster
                n_samples = enriched[clusters==cluster].shape[0]
                mask = (feature_count/n_samples >= 0.75)
                filtered_feats  = cluster_features[mask]
                filtered_counts = feature_count[mask]

                # Sort features by how much represented they are in the cluster
                sort_mask = np.argsort(filtered_counts)[::-1]

                representative_feats = filtered_feats[sort_mask]
                
                # Filter clusters that are noise or have no representative features
                if filtered_feats.size != 0 and cluster != -1:
                    clusters_of_interest.append(cluster)
                    cluster_feats.append(filtered_feats)
                    
                    string = list()
                    for feat in filtered_feats:
                        if feat in self.space_dict:
                            string.append(self.space_dict[feat])
                        else:
                            string.append(feat)
                    strings.append(string)
                            
                    coverage = [f"{n:.2f}%" for n in (filtered_counts[sort_mask]/n_samples)*100]
                    coverages.append(coverage)
                    
                    # Get molecule with higher membership probability ("centroid")
                    centroid = clusterer.weighted_cluster_centroid(cluster)
                    centroids.append(centroid)

                    
                # The resulting features are the label of that cluster
                cluster_label = [f in representative_feats for f in self.features]
                cluster_labels[cluster] = cluster_label
            
            centroids = np.vstack(centroids)
            
            if get_metrics:
                from chemicalchecker.tools.cc_charts.char_utils.plots import plot_metrics
            
                # Get prediction metrics for the cluster labels, before and after filtering
                # noise clusters
                predicted_labels = np.vstack([cluster_labels[cluster] for cluster in clusters])

                coverage = 1
                plot_metrics(enriched, 
                                     predicted_labels, 
                                     os.path.join(self.diags_path, 'unfiltered.png'), 
                                     ['Coverage', coverage],
                                     title='No filtering',
                                     color=cc_colors(self.dataset[:2])
                                    )

                filter_mask = np.array([cluster in clusters_of_interest for cluster in clusters])
                coverage = enriched[filter_mask].shape[0]/enriched.shape[0]

                plot_metrics(enriched[filter_mask], 
                                     predicted_labels[filter_mask], 
                                     os.path.join(self.diags_path, 'filtered.png'), 
                                     ['Coverage', coverage],
                                     title='Filtered',
                                     color=cc_colors(self.dataset[:2])
                                    )
            
            # Store the number of clusters
            n_clusters = len(clusters_of_interest)
            
            with open(os.path.join(self.stats_path, 'n_clusters.pkl'), 'wb') as fh:
                pickle.dump(n_clusters, fh)
                   
            # Barplot of the cluster sizes            
            fig, ax = plt.subplots(figsize=(11, 7))
            
            cluster_nums = [str(n + 1) for n in range(n_clusters)]
            cluster_sizes = [(clusters==cluster).sum() for cluster in clusters_of_interest]
            
            ax.bar(cluster_nums, 
                   cluster_sizes,
                    color=cc_colors(self.dataset[:2]),
                    alpha=0.4)
            
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of molecules')
            ax.set_title('Cluster sizes') 
            
            fig.savefig(os.path.join(self.diags_path, 'cluster_sizes.png'), dpi=300)
            
            # KDEplot of the cluster sizes
            fig, ax = plt.subplots(figsize=(7, 7))
            
            sns.kdeplot(cluster_sizes, 
                        color=cc_colors(self.dataset[:2]),
                        cut=0,
                        fill=True,
                        alpha=0.4,
                        ax=ax)
            
            ax.set_xlabel('Cluster size')
            ax.grid(True)
            
            fig.savefig(os.path.join(self.diags_path, 'cluster_sizes_kde.png'),
                        bbox_inches='tight',
                        dpi=300)
            
            with open(os.path.join(self.diags_path, 'sizes'), 'wb') as fh:
                pickle.dump(cluster_sizes, fh)
            
            # Save information about the clusters
            with open(os.path.join(self.diags_path, 'clusters.pkl'), 'wb') as fh:
                pickle.dump((centroids, cluster_nums, strings), fh)

            
            # Generate space chart
            fig, axs = plt.subplots(figsize=(10, 10))
            palette = make_cmap([(1, 1, 1), (0.1, 0.1, 0.1)])
            
            sns.kdeplot(x=safe_coords[:, 0],
                        y=safe_coords[:, 1],
                        fill=True,
                        cmap=palette,
                        levels=100,
                        antialiased=True,
                        alpha=0.6,
                        ax=ax)
            
            for idx, cluster in enumerate(clusters_of_interest):
                string = strings[idx]
                centroid = centroids[idx]
                
                try:
                    # Plot each one of the clusters
                    sns.kdeplot(x=safe_coords[clusters==cluster][:, 0],
                                y=safe_coords[clusters==cluster][:, 1],
                                levels=2, alpha=0.6, fill=True, ax=ax)
                    ax.text(centroid[0], centroid[1], s=idx + 1, fontsize=10)
                    ax.set_xlabel('tSNE 1')
                    ax.set_ylabel('tSNE 2')
                    ax.set_title(self.dataset)
                
                except Exception as e:
                    self.__log.warning(f'Representation of cluster {cluster} failed: {e}')

            
            # Add interactive elements
            scatter = ax.plot(centroids[:, 0], 
                              centroids[:, 1],  
                              'o', 
                              color='b',
                              mec='k', 
                              ms=30, 
                              mew=1, 
                              alpha=0, 
                              zorder=9999)
            
            source_path = os.path.dirname(__file__)
            css_path = os.path.join(source_path, '../tools/cc_charts/char_utils', 'tables.css')
            with open(css_path) as f:
                css = f.read()

            tables = list()
            
            for i in range(n_clusters):

                if not isinstance(strings[i], list):
                    strings[i] = [strings[i]]
                                    
                table = pd.DataFrame({'Feature': cluster_feats[i], 
                                      'Description': strings[i], 
                                      'Coverage': coverages[i]
                                     })
                table.index = ['']*table.shape[0]
                tables.append(str(table.to_html()))
                
            tooltip = mpld3.plugins.PointHTMLTooltip(scatter[0], 
                                                     labels=tables, 
                                                     voffset=10, 
                                                     hoffset=10, 
                                                     css=css)
            mpld3.plugins.connect(fig, tooltip)

            ax.set_aspect('equal')

            fig.set_facecolor('white')
            ax.grid(False)
            mpld3.save_html(fig, open(f'/aloy/home/amonsalve/visualizations/{self.dataset}_fig.html', 'w'))
            fig.savefig(os.path.join(self.diags_path, 'space_chart.png'), dpi=300, bbox_inches='tight')
            
            return fig
        
        self.__log.info('Cluster enrichment')
        clusterer, labels = cluster_enrichment()
        
        self.__log.info('Computing metrics for cluster labels')
        space_chart = compute_scores(clusterer, labels)
        
        with open(os.path.join(self.diags_path, 'space_chart.pkl'), 'wb') as fh:
            pickle.dump(space_chart, fh)
        
        with open(os.path.join(self.model_path, 'clusterer.pkl'), 'wb') as fh:
            pickle.dump(clusterer, fh)

        return clusterer
    
    def predict_feat(self, features, coords=None, mode=None):
        """Visualize a feature. Plots the tSNE projection of the molecules with available 
        signature 0 and a KDE (Kernel Density Estimate) of the molecules having the feature 
        of interest on top.
        
        Args:
            feature(str or list): feature(s) of interest.       
        """
        if not self.is_fit():
            self.__log.error('The visu instance has not been fitted yet. Run the fit() method first.')
            return
        
        # Transform the query to a list
        if not isinstance(features, list):
            features = [features]
        
        # Load the column of the experimental data corresponding to the query feature(s)
        with h5py.File(self.data_path) as f:
            feat_idx = np.nonzero([feature in features for feature in self.features])[0]
            V0 = f['V0'][:, feat_idx]
            enriched = f['enriched'][:, feat_idx]
            safe_coords = f['safe_coords'][:]
            
        # Load the background projection
        with open(os.path.join(self.diags_path, "safe_proj.pkl"), 'rb') as fh:
            fig = pickle.load(fh)
            ax = fig.get_axes()[0]
            
        handles = []
        
        colors = pick_colors(features)
        
        for idx, feature in enumerate(features):
            
            coords = safe_coords[enriched[:, idx]]
            title = f'{feature}: {self.space_dict[feature]}'

            color = mc.to_rgb(colors[idx])

            if mode == 'V0':

                coords = self.get_h5_dataset('safe_coords')[(V0[:, idx]==1)]
                ax.scatter(coords[:, 0],
                           coords[:, 1],
                           label=feature)
            
            elif mode == 'points' or len(coords) < 3:
                coords = self.get_h5_dataset('safe_coords')[(enriched[:, idx]==1)]
                ax.scatter(coords[:, 0],
                           coords[:, 1],
                           label=feature)  
            else:
                sns.kdeplot(x=coords[:, 0], 
                            y=coords[:, 1], 
                            ax=ax, 
                            levels=10, 
                            fill=True, 
                            alpha=0.4,
                            bw_adjust=0.5,
                            thresh=0.2,
                            color=color)

                handles.append(mpatches.Patch(color=color, label=title))
        
        ax.set_aspect('equal')
        ax.legend(handles=handles, loc='center right', bbox_to_anchor=(1.5, 0.5))
        ax.set_title(title)
        fig.set_facecolor('white')

        return fig
    
    def predict(self, query, kde=True, scatter=False):
        """Returns the inferred classes for a given molecule. It also plots the approximate
        location of the molecule in the space and a KDE representation of the inferred classes.
        
        Args:
            query(str): InChI key, name or SMILES of the molecule of interest.
            keytype(str): Type of query. Any of 'inchikey', 'name' or 'smiles'.
        """
        
        molname = query

        if not self.is_fit():
            self.__log.error('This visualization signature has not been fitted yet. Run the fit() method first.')
            return
        
        V1 = self.get_h5_dataset('V')
        coords = self.get_h5_dataset('safe_coords')
        radius = self.get_h5_attr('radius')        

        if self.shape[0] > 10000:
            random_idxs = np.sort(np.random.choice(
                            len(self.keys), 1000, replace=False))
            scores = self.get_h5_dataset('scores', mask=random_idxs)
        else:
            scores = self.get_h5_dataset('scores')
                
        # Get inchikey
        from chemicalchecker.tools.cc_charts.char_utils.util import query_to_inchikey
        self.__log.debug('Converting query to inchikey')
        keytype, inchikey = query_to_inchikey(query)
            
        # If working with signature 4 the molecule can be signaturized
        if self.cctype[-1] == '4':
            inchi = Converter().inchikey_to_inchi(inchikey)
            if self.dataset == 'P1.001':        
                signature = Signaturizer('E1').predict(inchi, keytype='InChI').signature
            else:
                signature = Signaturizer(self.dataset[:2]).predict(inchi, keytype='InChI').signature
            
        # Else check whether the molecule is present in that space    
        else:
            signature = V1[keys_V0==inchikey]          
            # If it is not present raise an error
            if len(signature) == 0:
                raise ValueError(f'Molecule not present in the signature {self.cctype[-1]} dataset')
        
        # Perform SAFE on the query molecule
        self.__log.debug('Performing SAFE')
        scorevec, positives, neigh_idxs = self.SAFE(signature)
        pred_class = self.features[positives]
        pred_scores = scorevec[positives]
         
        point_coords, _ = self.get_coords(scorevec.reshape(1, -1))
        
        # Order feature by their score
        order = np.argsort(pred_scores)[::-1]
        feats_to_plot = pred_class[order][:5]
                
        max_score = self.get_h5_attr('max_score')
        descriptions = [self.space_dict[c] for c in pred_class[order] if c in self.space_dict]
        res = pd.DataFrame(data=dict(Feature=pred_class[order], Description=descriptions, Score=[f'{a:.2f}' for a in pred_scores[order]/max_score]))
        
        with open(os.path.join(self.diags_path, 'safe_proj.pkl'), 'rb') as fh:
            fig = pickle.load(fh)
        
        ax = fig.get_axes()[0]

        colors = pick_colors(pred_class)
        colors = cm.tab10.colors
        
        handles = list()
        
        from mpld3 import plugins
        safe_coords = self.get_h5_dataset('safe_coords')
        ax.scatter(point_coords[:, 0], point_coords[:, 1], c='red', label=molname, zorder=9999)
        ax.scatter(safe_coords[neigh_idxs, 0], safe_coords[neigh_idxs, 1], c='gray', label='Neighbourhood', zorder=9998)
        
        collections = list()    
        for idx, feature in enumerate(feats_to_plot):
            
            # Allow to plot 5 areas maximum
            if idx > 4:
                break
            with h5py.File(self.data_path) as f:

                color = mc.to_rgb(colors[idx])

                feat_idx = np.argmax(self.features==feature)
                
                with h5py.File(self.data_path) as f:
                    enriched = f['enriched'][:, feat_idx]
                    
                n_enriched = enriched.sum()
                
                cmap = make_cmap((lighten_color(color, 1), lighten_color(color, 0.5)))
                
                label = self.space_dict[feature]
                
                if n_enriched > 3:
                    from numpy.linalg import LinAlgError
                    try:
                        plot = sns.kdeplot(x=safe_coords[:, 0][enriched],
                                    y=safe_coords[:, 1][enriched],
                                    fill=True,
                                    levels=2,
                                    color=color,
                                    thresh=0.2,
                                    bw_adjust=0.5,
                                    antialiased=False,
                                    alpha=0.6,
                                    ax=ax,
                                    label=label)
                        collections.append(plot.get_children()[0])
                    except LinAlgError:
                        ax.scatter(x=safe_coords[:, 0][enriched],
                               y=safe_coords[:, 1][enriched],
                               color=color)
                else:
                    ax.scatter(x=safe_coords[:, 0][enriched],
                               y=safe_coords[:, 1][enriched],
                               color=color)

                import matplotlib.patches as mpatches
                handles.append(mpatches.Patch(color=color, label=label))

        labels = [molname, 'Neighbourhood'] + [self.space_dict[feature] for feature in feats_to_plot]

        interactive_legend = plugins.InteractiveLegendPlugin(ax.get_legend_handles_labels()[0],
                                                 labels,
                                                 ax=ax,
                                                 alpha_unsel=0,
                                                 alpha_over=1.5, 
                                                 legend_offset=(-550, 0)
                                                 )
        plugins.connect(fig, interactive_legend)

        ax.set_xlabel('tSNE 1', fontsize=16)
        ax.set_ylabel('tSNE 2', fontsize=16)
        ax.grid(False)
        
        ax.set_xlim(point_coords[:, 0] - 10, point_coords[:, 0] + 10)
        ax.set_ylim(point_coords[:, 1] - 10, point_coords[:, 1] + 10)

        return fig, res
    
    def get_coords(self, signature, n_neighbors=5):
        scores = self.get_h5_dataset('scores')
        coords = self.get_h5_dataset('safe_coords')

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric=self.metric, n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(scores)
        dist, idx = nn.kneighbors(signature)

        idx = idx.flatten()
        dist = dist.flatten()

        neigh_keys = self.keys[idx]
        neigh_coords = coords[idx]
        weights = 1/dist
        weights[np.isinf(weights)] = weights[~np.isinf(weights)].max()

        new_coords = np.average(neigh_coords, 
                                axis=0, 
                                weights=weights)

        furthest = neigh_coords[np.argmax(dist)]
        deltas = np.abs(new_coords - furthest)

        return new_coords.reshape(-1, 2), deltas
    
    def molecule_boulder(self, query, keytype=None):
        
        with open(os.path.join(self.model_path, 'clusterer.pkl'), 'rb') as fh:
            clusterer = pickle.load(fh)                

        if not self.is_fit():
            self.__log.error('Visu signature is not fitted. Run the fit() method first.')
            return
        
        V1 = self.get_h5_dataset('V')
        
        coords = self.get_h5_dataset('safe_coords')
        
        with h5py.File(self.data_path) as f:
            thr = f['thr'][()]
        
        def get_coords(signature, n_neighbors=5):
            nn = NearestNeighbors(metric='euclidean', n_neighbors=n_neighbors)
            nn.fit(V1)
            dist, idx = nn.kneighbors(signature)
            new_coords = np.average(np.vstack([coords[self.keys==key] for key in self.keys[idx[0][1:]]]), 
                                    axis=0, 
                                    weights=1/dist[0][1:])
            
            fig, ax = plt.subplots()
            for i in idx:
                ax.scatter(coords[i, 0], coords[i, 1])
            ax.scatter(new_coords[0], new_coords[1])
            return new_coords.reshape(-1, 2)
        
        if keytype is None:
            from chemicalchecker.util.keytype import KeyTypeDetector
            
            kd = KeyTypeDetector('')
            keytype = kd.type(query)
            
            if keytype is None:
                keytype = 'name'
        
        # If input is a name
        if keytype == 'name':
            smi = Converter().chemical_name_to_smiles(query)
            query = Converter().smiles_to_inchi(smi)[0]

        
        # If input is a signature
        if isinstance(query, np.ndarray) or isinstance(query, list):
            signature = query
            point_coords = get_coords(signature)
             
        elif keytype=='smiles':
            signature = Signaturizer(self.dataset[:2]).predict(query, keytype='smiles').signature
            point_coords = get_coords(signature)
            
        # If working with signature 4 the molecule can be signaturized
        elif self.cctype[-1] == '4':
            inchi = Converter().inchikey_to_inchi(query)
            if self.dataset[:2]=='P1':
                self.dataset = 'E1.001'
            signature = Signaturizer(self.dataset[:2]).predict(inchi, keytype='InChI').signature
            point_coords = get_coords(signature)
            
        # Else check whether the molecule is present in that space    
        else:
            signature = V1[keys_V0==query]
            point_coords = coords[keys_V0==query]
            
            if len(signature) == 0:
                raise ValueError(f'Molecule not present in the signature {self.cctype[-1]} dataset')
                        
        scorevec = self.SAFE(signature)
        
        labels = clusterer.labels_

        cluster = approximate_predict(clusterer, point_coords)[0]
        
        with open(os.path.join(self.diags_path, 'space_chart.pkl'), 'rb') as fh:
            fig = pickle.load(fh)
        
        ax = fig.get_axes()[0]

        sns.kdeplot(coords[(labels==cluster), 0],
                    coords[(labels==cluster), 1],
                    levels=10,
                    alpha=0.4,
                    ax=ax
                   )
        return fig
    
    def SAFE(self, v, radius=None):
        V1 = self.get_h5_dataset('V')
        V0 = self.get_h5_dataset('V0')
        thr = self.get_h5_attr('thr')

        with h5py.File(self.data_path) as f:

            if radius is None:
                radius = self.get_h5_attr('radius')
            nn = NearestNeighbors(radius=radius, metric=self.metric, n_jobs=-1)
            nn.fit(V1)
            dist, neigh_idxs = nn.radius_neighbors(v)

            neigh_idxs = np.delete(neigh_idxs[0], (dist[0]<1e-5).nonzero()[0])

            scorevec = np.zeros_like(self.features, dtype='float64')
            
            v0 = V0[neigh_idxs]
            
            for feat_idx, feature in enumerate(tqdm(self.features)):
                space_size = len(V1)
                with_feature = V0[:, feat_idx].sum()
                
                a = (v0[:, feat_idx]!=0).sum()
                b = with_feature - a

                c = (v0[:, feat_idx]==0).sum()
                d = space_size - with_feature - c

                odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')

                score = -np.log10(p)
                scorevec[feat_idx] = score
                
        scorevec[scorevec > 1] = 1
        enriched = (scorevec >= thr)
        
        return scorevec, enriched, neigh_idxs
  
def pick_colors(items):
    if len(items) <= 10:
        colors = cm.tab10.colors

    elif len(items) <= 20:
        colors = cm.tab20.colors

    else:
        colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(items))]
    return colors

