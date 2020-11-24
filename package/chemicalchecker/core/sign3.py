"""Signature type 3.

Fixed-length (e.g. 128-d) representation of the data, capturing and inferring
the original (signature type 1) similarity of the data. Signatures type 3 are
available for any molecule of interest and have a confidence/applicability
measure assigned to them.
"""
import os
import h5py
import pickle
import numpy as np
from tqdm import tqdm
from time import time
from scipy import stats
from functools import partial
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, ks_2samp
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import robust_scale, StandardScaler

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import NeighborTripletTraintest
from chemicalchecker.util.splitter import Traintest
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class sign3(BaseSignature, DataSignature):
    """Signature type 3 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are:
                * 'sign2' for learning based on sign2
                * 'sign0' for learning based on sign0 (Morgan Fingerprint)
                * 'sign0_conf' for learning confidences based on MFP
                * 'prior' for learning prior in predictions
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               dataset, **params)
        self.data_path = os.path.join(self.signature_path, 'sign3.h5')
        DataSignature.__init__(self, self.data_path)
        # get parameters or default values
        self.params = dict()
        # parameters to learn from sign2
        default_sign2 = {
            'epochs': 10,
            'cpu': 8,
            'layers': ['Dense', 'Dense', 'Dense', 'Dense'],
            'layers_sizes': [1024, 512, 256, 128],
            'activations': ['selu', 'selu', 'selu', 'tanh'],
            'dropouts': [0.2, 0.2, 0.2, None],
            'learning_rate': 'auto',
            'batch_size': 128,
            'patience': 200,
            'loss_func': 'only_self_loss',
            'margin': 1.0,
            'alpha': 1.0,
            'num_triplets': 1000000,
            't_per': 0.01,
            'standard': False,
            'augment_fn': subsample,
            'augment_kwargs': {
                'dataset': [dataset],
            },
            'limit_mols': 100000
        }

        s1_ref = self.get_sign('sign1').get_molset("reference")
        opt_t_file = os.path.join(s1_ref.model_path, "opt_t.h5")
        try:
            opt_t = DataSignature(opt_t_file).get_h5_dataset('opt_t')
            default_sign2.update({'t_per': opt_t})
            self.t_per = opt_t
        except Exception as ex:
            self.__log.warning('Failed setting opt_t: %s' % str(ex))
            self.t_per = 0.01

        default_sign2.update(params.get('sign2', {}))
        self.params['sign2_lr'] = default_sign2.copy()
        self.params['sign2'] = default_sign2
        # parameters to learn from sign0
        default_sign0 = {
            'epochs': 5,
            'cpu': 8,
            'learning_rate': 1e-3,
            'layers': ['Dense', 'Dense', 'Dense', 'Dense'],
            'layers_sizes': [1024, 512, 256, 128],
            'activations': ['selu', 'selu', 'selu', 'tanh'],
            'dropouts': [0.2, 0.2, 0.2, None],
            'batch_size': 128,
            'num_triplets': 100000,
            'margin': 1.0,
            'alpha': 0.5,
        }
        # default_sign0.update(params.get('sign0', {}))
        self.params['sign0'] = default_sign0
        # parameters to learn confidence from sign0
        default_sign0_conf = {
        }
        default_sign0_conf.update(params.get('sign0_conf', {}))
        self.params['sign0_conf'] = default_sign0_conf

    @staticmethod
    def save_sign2_universe(sign2_list, destination):
        """Create a file with all signatures 2 for each molecule in the CC.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where the H5 is saved.
        """
        # get sorted universe inchikeys and CC signatures
        sign3.__log.info("Generating signature 2 universe matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign3.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, 128 * tot_ds),
                              dtype=np.float32)
            fh.create_dataset('keys', data=np.array(
                inchikeys, DataSignature.string_dtype()))
            for idx, sign in enumerate(sign2_list):
                sign3.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                _, vectors = sign.get_vectors(inchikeys, include_nan=True)
                fh['x_test'][:, idx * 128:(idx + 1) * 128] = vectors
                del vectors

    @staticmethod
    def save_sign2_coverage(sign2_list, destination):
        """Create a file with all signatures 2 coverage of molecule in the CC.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            destination(str): Path where the H5 is saved.
        """
        # get sorted universe inchikeys and CC signatures
        sign3.__log.info("Generating signature 2 coverage matrix.")
        inchikeys = set()
        for sign in sign2_list:
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        tot_inks = len(inchikeys)
        tot_ds = len(sign2_list)
        sign3.__log.info("Saving coverage for %s dataset and %s molecules." %
                         (tot_ds, tot_inks))
        # build matrix stacking horizontally signature
        if os.path.isfile(destination):
            sign3.__log.warning("Skipping as destination %s already exists." %
                                destination)
            return
        with h5py.File(destination, "w") as fh:
            fh.create_dataset('x_test', (tot_inks, tot_ds), dtype=np.float32)
            fh.create_dataset('keys', data=np.array(
                inchikeys, DataSignature.string_dtype()))
            for idx, sign in enumerate(sign2_list):
                sign3.__log.info("Fetching from %s" % sign.data_path)
                # including NaN we have the correct number of molecules
                coverage = np.isin(
                    list(inchikeys), list(sign.keys), assume_unique=True)
                sign3.__log.info("%s has %s Signature 2." %
                                 (sign.dataset, np.count_nonzero(coverage)))
                fh['x_test'][:, idx:(idx + 1)] = np.expand_dims(coverage, 1)

    def save_sign2_matrix(self, destination):
        """Save matrix of pairs of horizontally stacked signature 2.

        This is the matrix for training the signature 3. It is defined for all
        molecules or which we have a signature 2 in the current space.
        It's a subset of the universe of stacked sign2 file.

        Args:
            destination(str): Path where to save the matrix (HDF5 file).
        """
        self.__log.debug('Saving sign2 traintest to: %s' % destination)
        universe = DataSignature(self.sign2_universe)
        mask = np.isin(list(universe.keys), list(self.sign2_self.keys))
        universe.make_filtered_copy(destination, mask)
        # rename the dataset to what the splitter expects
        with h5py.File(destination, 'a') as hf:
            hf['x'] = hf['x_test']
            del hf['x_test']

    def learn_sign2(self, params, reuse=True, suffix=None, evaluate=True):
        """Learn the signature 3 model.

        This method is used twice. First to evaluate the performances of the
        Siamese model. Second to train the final model on the full set of data.

        Args:
            params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 2 matrix).
            suffix(str): A suffix for the Siamese model path (e.g.
                'sign3/models/siamese_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
        """
        try:
            from chemicalchecker.tool.siamese import SiameseTriplets
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")
        # get params and set folder
        self.update_status("Training %s model" % suffix)
        self.__log.debug('Siamese suffix %s' % suffix)
        if suffix:
            siamese_path = os.path.join(self.model_path, 'siamese_%s' % suffix)
            traintest_file = os.path.join(self.model_path,
                                          'traintest_%s.h5' % suffix)
            params['traintest_file'] = params.get(
                'traintest_file', traintest_file)
        else:
            siamese_path = os.path.join(self.model_path, 'siamese')
        if 'model_dir' in params:
            siamese_path = params.get('model_dir')
        if not reuse or not os.path.isdir(siamese_path):
            os.makedirs(siamese_path)
        # generate input matrix
        sign2_matrix = os.path.join(self.model_path, 'train.h5')
        if not reuse or not os.path.isfile(sign2_matrix):
            self.save_sign2_matrix(sign2_matrix)
        # if evaluating, perform the train-test split
        traintest_file = params.get('traintest_file')
        X = DataSignature(sign2_matrix)
        if evaluate:
            num_triplets = params.get('num_triplets', 1e6)
            cpu = params.get('cpu', 1)
            if not reuse or not os.path.isfile(traintest_file):
                NeighborTripletTraintest.create(
                    X, traintest_file, self.neig_sign,
                    split_names=['train', 'test'],
                    split_fractions=[.8, .2],
                    suffix=suffix,
                    num_triplets=num_triplets,
                    t_per=params['t_per'],
                    cpu=cpu,
                    limit=params['limit_mols'])
        else:
            num_triplets = params.get('num_triplets', 1e6)
            cpu = params.get('cpu', 1)
            if not reuse or not os.path.isfile(traintest_file):
                NeighborTripletTraintest.create(
                    X, traintest_file, self.neig_sign,
                    split_names=['train'],
                    split_fractions=[1.0],
                    suffix=suffix,
                    num_triplets=num_triplets,
                    t_per=params['t_per'],
                    cpu=cpu,
                    limit=params['limit_mols'])
        # update the subsampling parameter
        if 'augment_kwargs' in params:
            ds = params['augment_kwargs']['dataset']
            dataset_idx = np.argwhere(np.isin(self.src_datasets, ds)).flatten()
            if len(dataset_idx) > 1:
                self.__log.warning('Dataset %s is repeated' % ds)
            dataset_idx = np.array(dataset_idx[:1])
            # compute probabilities for subsampling
            trim_mask, p_nr_unknown, p_keep_unknown, p_nr_known, p_keep_known = subsampling_probs(
                self.sign2_coverage, dataset_idx)
            trim_dataset_idx = np.argwhere(
                np.arange(len(trim_mask))[trim_mask] == dataset_idx).ravel()[0]
            params['augment_kwargs']['p_nr'] = (p_nr_unknown, p_nr_known)
            params['augment_kwargs']['p_keep'] = (p_keep_unknown, p_keep_known)
            params['augment_kwargs']['dataset_idx'] = [trim_dataset_idx]
            params['augment_kwargs']['p_only_self'] = 0.0
            params['trim_mask'] = trim_mask
        # train siamese network
        self.__log.debug('Siamese training on %s' % traintest_file)
        siamese = SiameseTriplets(siamese_path, evaluate=evaluate, **params)
        siamese.fit()
        self.__log.debug('model saved to: %s' % siamese_path)
        # if final we are done
        if not evaluate:
            return siamese
        # save validation plots
        self.plot_validations(siamese, dataset_idx, traintest_file)
        # when evaluating also save prior and confidence models
        prior_model, prior_sign_model, confidence_model = self.train_confidence(
            traintest_file, X, suffix, siamese)
        # update the parameters with the new nr_of epochs and lr
        self.params['sign2']['epochs'] = siamese.last_epoch
        return siamese, prior_model, prior_sign_model, confidence_model

    def train_confidence(self, traintest_file, X, suffix, siamese,
                         max_x=10000, max_neig=50000, p_self=0.0):
        """Train confidence and prior models."""
        # get sorted keys from siamese traintest file
        self.update_status('Training applicability')
        tt = DataSignature(traintest_file)
        test_inks = tt.get_h5_dataset('keys_test')[:max_x]
        test_inks = np.sort(test_inks)
        train_inks = tt.get_h5_dataset('keys_train')[:max_neig]
        train_inks = np.sort(train_inks)
        test_mask = np.isin(list(self.sign2_self.keys), list(test_inks),
                            assume_unique=True)
        train_mask = np.isin(list(self.sign2_self.keys), list(train_inks),
                             assume_unique=True)
        # confidence is going to be trained only on siamese test data
        confidence_train_x = X.get_h5_dataset('x', mask=test_mask)
        s2_test = self.sign2_self.get_h5_dataset('V', mask=test_mask)
        s2_test_x = confidence_train_x[:, self.dataset_idx[0]
                                       * 128: (self.dataset_idx[0] + 1) * 128]
        assert(np.all(s2_test == s2_test_x))
        # siamese train is going to be used for appticability domain
        known_x = X.get_h5_dataset('x', mask=train_mask)
        # generate train-test split for confidence estimation
        split_names = ['train', 'test']
        split_fractions = [0.8, 0.2]
        split_idxs = Traintest.get_split_indeces(
            confidence_train_x.shape[0], split_fractions, random_state=42)
        splits = list(zip(split_names, split_fractions, split_idxs))

        # train prior model
        prior_path = os.path.join(self.model_path, 'prior_%s' % suffix)
        os.makedirs(prior_path, exist_ok=True)
        prior_model = self.train_prior_model(siamese, confidence_train_x,
                                             splits, prior_path,
                                             max_x=max_x, p_self=p_self)

        # train prior signature model
        prior_sign_path = os.path.join(self.model_path,
                                       'prior_sign_%s' % suffix)
        os.makedirs(prior_sign_path, exist_ok=True)
        prior_sign_model = self.train_prior_signature_model(
            siamese, confidence_train_x, splits, prior_sign_path,
            max_x=max_x, p_self=p_self)

        # train confidence model
        confidence_path = os.path.join(self.model_path,
                                       'confidence_%s' % suffix)
        os.makedirs(confidence_path, exist_ok=True)
        confidence_model = self.train_confidence_model(
            siamese, known_x, confidence_train_x, splits,
            prior_model, prior_sign_model,
            confidence_path, p_self=p_self)
        return prior_model, prior_sign_model, confidence_model

    def rerun_confidence(self, cc, suffix, train=True, update_sign=True, chunk_size=10000, sign2_universe=None, sign2_coverage=None):
        """Rerun confidence trainining and estimation"""
        try:
            import faiss
            from chemicalchecker.tool.siamese import SiameseTriplets
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")
        sign1_self = cc.get_signature("sign1", "full", self.dataset)
        sign2_self = cc.get_signature("sign2", "full", self.dataset)
        sign2_list = [cc.get_signature("sign2", "full", d)
                      for d in cc.datasets_exemplary()]
        if sign2_universe is None:
            sign2_universe = os.path.join(cc.cc_root, 'full', 'all_sign2.h5')
        if sign2_coverage is None:
            sign2_coverage = os.path.join(
                cc.cc_root, 'full', 'all_sign2_coverage.h5')

        self.src_datasets = [sign.dataset for sign in sign2_list]
        self.neig_sign = sign1_self
        self.sign2_self = sign2_self
        self.sign2_list = sign2_list
        self.sign2_coverage = sign2_coverage
        self.sign2_universe = sign2_universe
        self.dataset_idx = np.argwhere(
            np.isin(self.src_datasets, self.dataset)).flatten()

        siamese_path = os.path.join(self.model_path, 'siamese_%s' % suffix)
        siamese = SiameseTriplets(siamese_path, predict_only=True)
        siamese_cp = SiameseTriplets(siamese.model_dir, predict_only=True)

        if train:
            traintest_file = os.path.join(
                self.model_path, 'traintest_%s.h5' % suffix)
            X = DataSignature(os.path.join(self.model_path, 'train.h5'))
            prior_mdl, prior_sign_mdl, conf_mdl = self.train_confidence(
                traintest_file, X, suffix, siamese)
            if not update_sign:
                return
        else:
            # part of confidence is the priors
            prior_path = os.path.join(self.model_path, 'prior_eval')
            prior_file = os.path.join(prior_path, 'prior.pkl')
            prior_mdl = pickle.load(open(prior_file, 'rb'))

            # part of confidence is the priors based on signatures
            prior_sign_path = os.path.join(
                self.model_path, 'prior_sign_eval')
            prior_sign_file = os.path.join(prior_sign_path, 'prior.pkl')
            prior_sign_mdl = pickle.load(open(prior_sign_file, 'rb'))

            # and finally the linear combination of scores
            confidence_path = os.path.join(self.model_path, 'confidence_eval')
            confidence_file = os.path.join(
                confidence_path, 'confidence.pkl')
            calibration_file = os.path.join(
                confidence_path, 'calibration.pkl')
            conf_mdl = (pickle.load(open(confidence_file, 'rb')),
                        pickle.load(open(calibration_file, 'rb')))

        # another part of confidence is the applicability
        confidence_path = os.path.join(self.model_path, 'confidence_eval')
        neig_file = os.path.join(confidence_path, 'neig.index')
        app_neig = faiss.read_index(neig_file)
        known_dist = os.path.join(confidence_path, 'known_dist.h5')
        app_range = DataSignature(known_dist).get_h5_dataset(
            'applicability_range')
        _, trim_mask = self.realistic_subsampling_fn()

        # get sorted universe inchikeys
        self.universe_inchikeys = self.get_universe_inchikeys()
        tot_inks = len(self.universe_inchikeys)
        known_mask = np.isin(list(self.universe_inchikeys),
                             list(self.sign2_self.keys),
                             assume_unique=True)

        with h5py.File(self.data_path, "a") as results:
            # the actual confidence value will be stored here
            safe_create(results, 'confidence', (tot_inks,),
                        dtype=np.float32)
            # this is to store robustness
            safe_create(results, 'robustness',
                        (tot_inks,), dtype=np.float32)
            # this is to store applicability
            safe_create(results, 'applicability',
                        (tot_inks,), dtype=np.float32)
            # this is to store priors
            safe_create(results, 'prior',
                        (tot_inks,), dtype=np.float32)
            # this is to store priors based on signature
            safe_create(results, 'prior_signature',
                        (tot_inks,), dtype=np.float32)

            # predict signature 3 for universe molecules
            with h5py.File(sign2_universe, "r") as features:
                # reference prediction (based on no information)
                nan_feat = np.full((1, features['x_test'].shape[1]),
                                   np.nan, dtype=np.float32)
                nan_pred = siamese.predict(nan_feat)
                # read input in chunks
                for idx in tqdm(range(0, tot_inks, chunk_size),
                                desc='Predicting'):
                    chunk = slice(idx, idx + chunk_size)
                    feat = features['x_test'][chunk]
                    # save confidence natural scores
                    # compute prior from coverage
                    cov = ~np.isnan(feat[:, 0::128])
                    prior = prior_mdl.predict(cov[:, trim_mask])
                    results['prior'][chunk] = prior
                    # and from prediction
                    preds = siamese.predict(feat)
                    prior_sign = prior_sign_mdl.predict(preds)
                    results['prior_signature'][chunk] = prior_sign
                    # conformal prediction
                    ints, robs, cons = self.conformal_prediction(
                        siamese_cp, feat, nan_pred=nan_pred)
                    results['robustness'][chunk] = robs
                    # distance from known predictions
                    app, centrality, _ = self.applicability_domain(
                        app_neig, feat, siamese, app_range=app_range,
                        n_samples=1)
                    results['applicability'][chunk] = app
                    # and estimate confidence
                    conf_feats = np.vstack(
                        [app, robs, prior, prior_sign, ints]).T
                    conf_estimate = conf_mdl[0].predict(conf_feats)
                    conf_calib = conf_mdl[1].predict(
                        np.expand_dims(conf_estimate, 1))
                    results['confidence'][chunk] = conf_calib
                # conpute confidence where self is known
                known_idxs = np.argwhere(known_mask).flatten()
                # iterate on chunks of knowns
                for idx in tqdm(range(0, len(known_idxs), 10000),
                                desc='Computing Confidence'):
                    chunk = slice(idx, idx + 10000)
                    feat = features['x_test'][known_idxs[chunk]]
                    # predict with all features
                    preds_all = siamese.predict(feat)
                    # predict with only-self features
                    feat_onlyself = mask_keep(self.dataset_idx, feat)
                    preds_onlyself = siamese.predict(feat_onlyself)
                    # confidence is correlation ALL vs. ONLY-SELF
                    corrs = row_wise_correlation(
                        preds_onlyself, preds_all, scaled=True)
                    results['confidence'][known_idxs[chunk]] = corrs

    def realistic_subsampling_fn(self):
        # realistic subsampling function
        trim_mask, p_nr_unk, p_keep_unk, p_nr_kno, p_keep_kno = \
            subsampling_probs(self.sign2_coverage, self.dataset_idx)
        p_nr = (p_nr_unk, p_nr_kno)
        p_keep = (p_keep_unk, p_keep_kno)
        trim_dataset_idx = np.argwhere(
            np.arange(len(trim_mask))[trim_mask] == self.dataset_idx).ravel()[0]
        realistic_fn = partial(subsample, p_only_self=0.0, p_self=0.0,
                               dataset_idx=trim_dataset_idx,
                               p_nr=p_nr, p_keep=p_keep)
        return realistic_fn, trim_mask

    def train_prior_model(self, siamese, train_x, splits, save_path,
                          max_x=10000, n_samples=5, p_self=0.0, plots=True):
        """Train prior predictor."""
        def get_weights(y, p=2):
            h, b = np.histogram(y, 20)
            b = [np.mean([b[i], b[i + 1]]) for i in range(0, len(h))]
            w = np.interp(y, b, h).ravel()
            w = -(w / np.sum(w)) + 1e-10
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            w = w**p
            return w

        def histograms(ax, yp, yt, title):
            ax.hist(yp, 10, range=(-1, 1), color="red",
                    label="Pred", alpha=0.5)
            ax.hist(yt, 10, range=(-1, 1), color="blue",
                    label="True", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Value")
            ax.set_ylabel("Counts")
            ax.set_title(title)

        def scatter(ax, yp, yt, joint_lim=True):
            x = yp
            y = yt
            xy = np.vstack([x, y])
            try:
                z = gaussian_kde(xy)(xy)
            except Exception as ex:
                self.__log.warning(str(ex))
                z = np.ones(x.shape)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=10, edgecolor='')
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if joint_lim:
                lim = (np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]]))
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.plot([lim[0], lim[1]], [lim[0], lim[1]],
                        color="gray", ls='--', lw=1)
            slope, intercept, r, p_val, stde = stats.linregress(x, y)
            line = slope * x + intercept
            ax.plot(x, line, 'r',
                    label='y={:.2f}x+{:.2f}'.format(slope, intercept))
            title = "rho = %.2f" % pearsonr(x, y)[0]
            ax.set_title(title)
            ax.legend()

        def importances(ax, mod, trim_mask):
            from chemicalchecker.util.plot import coord_color
            y = mod.feature_importances_
            datasets = ["%s%s" % (x, y) for x in "ABCDE" for y in "12345"]
            if trim_mask is not None:
                datasets = np.array(datasets)[trim_mask]
            else:
                datasets = np.array(datasets)
            x = np.array([i for i in range(0, len(datasets))])
            idxs = np.argsort(y)
            datasets = datasets[idxs]
            y = y[idxs]
            colors = [coord_color(ds) for ds in datasets]
            ax.scatter(y, x, color=colors)
            ax.set_xlabel("Importance")
            ax.set_yticks(x)
            ax.set_yticklabels(datasets)
            ax.set_title("Importance")
            ax.axvline(0, color="red", lw=1)

        def analyze(mod, x_tr, y_tr, x_te, y_te, trim_mask):
            import matplotlib.pyplot as plt
            y_tr_p = mod.predict(x_tr)
            y_te_p = mod.predict(x_te)
            plt.close('all')
            fig = plt.figure(constrained_layout=True, figsize=(8, 6))
            gs = fig.add_gridspec(2, 3)
            ax = fig.add_subplot(gs[0, 0])
            histograms(ax, y_tr_p, y_tr, "Train")
            ax = fig.add_subplot(gs[0, 1])
            histograms(ax, y_te_p, y_te, "Test")
            ax = fig.add_subplot(gs[1, 0])
            scatter(ax, y_tr_p, y_tr)
            ax = fig.add_subplot(gs[1, 1])
            scatter(ax, y_te_p, y_te)
            ax = fig.add_subplot(gs[0:2, 2])
            importances(ax, mod, trim_mask)
            if plots:
                plt.savefig(os.path.join(save_path, 'prior_stats.png'))
                plt.close()

        def find_p(mod, x_tr, y_tr, x_te, y_te):
            import matplotlib.pyplot as plt
            test = ks_2samp
            ps = []
            ss_te = []
            for p in np.linspace(0, 3, 10):
                w = get_weights(y_tr, p=p)
                mod.fit(x_tr, y_tr, sample_weight=w)
                y_te_p = mod.predict(x_te)
                s_te = test(y_te, y_te_p)[0]
                ps += [p]
                ss_te += [s_te]
            p = ps[np.argmin(ss_te)]
            if plots:
                plt.close('all')
                plt.scatter(ps, ss_te)
                plt.title("%.2f" % p)
                plt.savefig(os.path.join(save_path, 'prior_p.png'))
                plt.close()
            return p

        self.__log.info('Training PRIOR model')
        # define subsampling
        realistic_fn, trim_mask = self.realistic_subsampling_fn()
        # generate train test split
        out_file = os.path.join(save_path, 'data.h5')
        with h5py.File(out_file, "w") as fh:
            for split_name, split_frac, split_idx in splits:
                split_x = train_x[split_idx]
                split_total_x = int(max_x * split_frac)
                available_x = split_x.shape[0]
                X = np.zeros((split_total_x, np.sum(trim_mask)))
                Y = np.zeros((split_total_x, 1))
                preds_onlyselfs = np.zeros((split_total_x, 128))
                preds_noselfs = np.zeros((split_total_x, 128))
                feats = np.zeros((split_total_x, 3200))
                # prepare X and Y in chunks
                chunk_size = max(10000, int(np.floor(available_x / 10)))
                reached_max = False
                for i in range(0, int(np.ceil(split_total_x / available_x))):
                    for idx in range(0, available_x, chunk_size):
                        # define source chunk
                        src_start = idx
                        src_end = idx + chunk_size
                        if src_end > available_x:
                            src_end = available_x
                        # define destination chunk
                        dst_start = src_start + (int(available_x) * i)
                        dst_end = src_end + (available_x * i)
                        if dst_end > split_total_x:
                            dst_end = split_total_x
                            reached_max = True
                            src_end = dst_end - (int(available_x) * i)
                        src_chunk = slice(src_start, src_end)
                        dst_chunk = slice(dst_start, dst_end)
                        # get only-self and not-self predictions
                        feat = split_x[src_chunk]
                        feats[dst_chunk] = feat
                        preds_onlyself = siamese.predict(
                            mask_keep(self.dataset_idx, feat))
                        preds_onlyselfs[dst_chunk] = preds_onlyself
                        preds_noself = siamese.predict(
                            mask_exclude(self.dataset_idx, feat))
                        preds_noselfs[dst_chunk] = preds_noself
                        # the prior is only-self vs not-self predictions
                        corrs = row_wise_correlation(
                            preds_onlyself, preds_noself, scaled=True)
                        Y[dst_chunk] = np.expand_dims(corrs, 1)
                        # the X is the dataset presence in the not-self
                        presence = ~np.isnan(feat[:, ::128])[:, trim_mask]
                        X[dst_chunk] = presence.astype(int)
                        # check if enought
                        if reached_max:
                            break
                variables = [X, Y, feat, preds_onlyselfs, preds_noselfs]
                names = ['x', 'y', 'feat', 'preds_onlyselfs', 'preds_noselfs']
                for var, name in zip(variables, names):
                    ds_name = '%s_%s' % (name, split_name)
                    self.__log.debug('writing %s: %s' % (ds_name, var.shape))
                    fh.create_dataset(ds_name, data=var)
        traintest = DataSignature(out_file)
        x_tr = traintest.get_h5_dataset('x_train')
        y_tr = traintest.get_h5_dataset('y_train').ravel()
        x_te = traintest.get_h5_dataset('x_test')
        y_te = traintest.get_h5_dataset('y_test').ravel()
        # fit model
        model = RandomForestRegressor(n_estimators=1000, max_features=None,
                                      min_samples_leaf=0.01, n_jobs=4)
        p = find_p(model, x_tr, y_tr, x_te, y_te)
        model.fit(x_tr, y_tr, sample_weight=get_weights(y_tr, p=p))
        if plots:
            analyze(model, x_tr, y_tr, x_te, y_te, trim_mask)
        predictor_path = os.path.join(save_path, 'prior.pkl')
        pickle.dump(model, open(predictor_path, 'wb'))
        return model

    def train_prior_signature_model(self, siamese, train_x, splits,
                                    save_path, max_x=10000, n_samples=5,
                                    p_self=0.0, plots=True):
        """Train prior predictor."""
        def get_weights(y, p=2):
            h, b = np.histogram(y, 20)
            b = [np.mean([b[i], b[i + 1]]) for i in range(0, len(h))]
            w = np.interp(y, b, h).ravel()
            w = -(w / np.sum(w)) + 1e-10
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            w = w**p
            return w

        def histograms(ax, yp, yt, title):
            ax.hist(yp, 10, range=(-1, 1), color="red",
                    label="Pred", alpha=0.5)
            ax.hist(yt, 10, range=(-1, 1), color="blue",
                    label="True", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Value")
            ax.set_ylabel("Counts")
            ax.set_title(title)

        def scatter(ax, yp, yt, joint_lim=True):
            x = yp
            y = yt
            xy = np.vstack([x, y])
            try:
                z = gaussian_kde(xy)(xy)
            except Exception as ex:
                self.__log.warning(str(ex))
                z = np.ones(x.shape)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=10, edgecolor='')
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if joint_lim:
                lim = (np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]]))
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.plot([lim[0], lim[1]], [lim[0], lim[1]],
                        color="gray", ls='--', lw=1)
            slope, intercept, r, p_val, stde = stats.linregress(x, y)
            line = slope * x + intercept
            ax.plot(x, line, 'r',
                    label='y={:.2f}x+{:.2f}'.format(slope, intercept))
            title = "rho = %.2f" % pearsonr(x, y)[0]
            ax.set_title(title)
            ax.legend()

        def importances(ax, mod):
            y = mod.feature_importances_
            datasets = np.arange(128)
            datasets = np.array(datasets)
            x = np.array([i for i in range(0, len(datasets))])
            idxs = np.argsort(y)
            datasets = datasets[idxs]
            y = y[idxs]
            ax.scatter(y, x)
            ax.set_xlabel("Importance")
            ax.set_yticks(x)
            ax.set_yticklabels(datasets)
            ax.set_title("Importance")
            ax.axvline(0, color="red", lw=1)

        def analyze(mod, x_tr, y_tr, x_te, y_te):
            import matplotlib.pyplot as plt
            y_tr_p = mod.predict(x_tr)
            y_te_p = mod.predict(x_te)
            plt.close('all')
            fig = plt.figure(constrained_layout=True, figsize=(8, 6))
            gs = fig.add_gridspec(2, 3)
            ax = fig.add_subplot(gs[0, 0])
            histograms(ax, y_tr_p, y_tr, "Train")
            ax = fig.add_subplot(gs[0, 1])
            histograms(ax, y_te_p, y_te, "Test")
            ax = fig.add_subplot(gs[1, 0])
            scatter(ax, y_tr_p, y_tr)
            ax = fig.add_subplot(gs[1, 1])
            scatter(ax, y_te_p, y_te)
            ax = fig.add_subplot(gs[0:2, 2])
            importances(ax, mod)
            if plots:
                plt.savefig(os.path.join(save_path, 'prior_stats.png'))
                plt.close()

        def find_p(mod, x_tr, y_tr, x_te, y_te):
            import matplotlib.pyplot as plt
            test = ks_2samp
            ps = []
            ss_te = []
            for p in np.linspace(0, 3, 10):
                w = get_weights(y_tr, p=p)
                mod.fit(x_tr, y_tr, sample_weight=w)
                y_te_p = mod.predict(x_te)
                s_te = test(y_te, y_te_p)[0]
                ps += [p]
                ss_te += [s_te]
            p = ps[np.argmin(ss_te)]
            if plots:
                plt.close('all')
                plt.scatter(ps, ss_te)
                plt.title("%.2f" % p)
                plt.savefig(os.path.join(save_path, 'prior_p.png'))
                plt.close()
            return p

        self.__log.info('Training PRIOR SIGNATURE model')
        # define subsampling
        realistic_fn, trim_mask = self.realistic_subsampling_fn()
        # generate train test split
        out_file = os.path.join(save_path, 'data.h5')
        with h5py.File(out_file, "w") as fh:
            for split_name, split_frac, split_idx in splits:
                split_x = train_x[split_idx]
                split_total_x = int(max_x * split_frac)
                available_x = split_x.shape[0]
                X = np.zeros((split_total_x, 128))
                Y = np.zeros((split_total_x, 1))
                preds_onlyselfs = np.zeros((split_total_x, 128))
                preds_noselfs = np.zeros((split_total_x, 128))
                feats = np.zeros((split_total_x, 3200))
                # prepare X and Y in chunks
                chunk_size = max(10000, int(np.floor(available_x / 10)))
                reached_max = False
                for i in range(0, int(np.ceil(split_total_x / available_x))):
                    for idx in range(0, available_x, chunk_size):
                        # define source chunk
                        src_start = idx
                        src_end = idx + chunk_size
                        if src_end > available_x:
                            src_end = available_x
                        # define destination chunk
                        dst_start = src_start + (int(available_x) * i)
                        dst_end = src_end + (available_x * i)
                        if dst_end > split_total_x:
                            dst_end = split_total_x
                            reached_max = True
                            src_end = dst_end - (int(available_x) * i)
                        src_chunk = slice(src_start, src_end)
                        dst_chunk = slice(dst_start, dst_end)
                        # get only-self and not-self predictions
                        feat = split_x[src_chunk]
                        feats[dst_chunk] = feat
                        preds_onlyself = siamese.predict(
                            mask_keep(self.dataset_idx, feat))
                        preds_onlyselfs[dst_chunk] = preds_onlyself
                        preds_noself = siamese.predict(
                            mask_exclude(self.dataset_idx, feat))
                        preds_noselfs[dst_chunk] = preds_noself
                        # the prior is only-self vs not-self predictions
                        corrs = row_wise_correlation(
                            preds_onlyself, preds_noself, scaled=True)
                        Y[dst_chunk] = np.expand_dims(corrs, 1)
                        X[dst_chunk] = preds_noself
                        # check if enought
                        if reached_max:
                            break
                variables = [X, Y, feat, preds_onlyselfs, preds_noselfs]
                names = ['x', 'y', 'feat', 'preds_onlyselfs', 'preds_noselfs']
                for var, name in zip(variables, names):
                    ds_name = '%s_%s' % (name, split_name)
                    self.__log.debug('writing %s: %s' % (ds_name, var.shape))
                    fh.create_dataset(ds_name, data=var)
        traintest = DataSignature(out_file)
        x_tr = traintest.get_h5_dataset('x_train')
        y_tr = traintest.get_h5_dataset('y_train').ravel()
        x_te = traintest.get_h5_dataset('x_test')
        y_te = traintest.get_h5_dataset('y_test').ravel()
        # fit model
        model = RandomForestRegressor(n_estimators=1000, max_features='sqrt',
                                      min_samples_leaf=0.01, n_jobs=4)
        p = find_p(model, x_tr, y_tr, x_te, y_te)
        model.fit(x_tr, y_tr, sample_weight=get_weights(y_tr, p=p))
        if plots:
            analyze(model, x_tr, y_tr, x_te, y_te)
        predictor_path = os.path.join(save_path, 'prior.pkl')
        pickle.dump(model, open(predictor_path, 'wb'))
        return model

    def train_confidence_model(self, siamese, neig_x, train_x, splits,
                               prior_model, prior_sign_model, save_path,
                               p_self=0.0, plots=True):
        # save linear model combining confidence natural scores

        def get_weights(y, p=2):
            h, b = np.histogram(y, 20)
            b = [np.mean([b[i], b[i + 1]]) for i in range(0, len(h))]
            w = np.interp(y, b, h).ravel()
            w = -(w / np.sum(w)) + 1e-10
            w = (w - np.min(w)) / (np.max(w) - np.min(w))
            w = w**p
            return w

        def histograms(ax, yp, yt, title):
            ax.hist(yp, 10, range=(-1, 1), color="red",
                    label="Pred", alpha=0.5)
            ax.hist(yt, 10, range=(-1, 1), color="blue",
                    label="True", alpha=0.5)
            ax.legend()
            ax.set_xlabel("Value")
            ax.set_ylabel("Counts")
            ax.set_xlim((-1, 1))
            ax.set_title(title)

        def scatter(ax, yp, yt, joint_lim=True):
            x = yp
            y = yt
            xy = np.vstack([x, y])
            try:
                z = gaussian_kde(xy)(xy)
            except Exception as ex:
                self.__log.warning(str(ex))
                z = np.ones(x.shape)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=10, edgecolor='')
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if joint_lim:
                lim = (np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]]))
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.plot([lim[0], lim[1]], [lim[0], lim[1]],
                        color="gray", ls='--', lw=1)
            slope, intercept, r, p_val, stde = stats.linregress(x, y)
            line = slope * x + intercept
            ax.plot(x, line, 'r',
                    label='y={:.2f}x+{:.2f}'.format(slope, intercept))
            title = "rho = %.2f" % pearsonr(x, y)[0]
            ax.set_title(title)
            ax.legend()

        def analyze(mod, cal, x_tr, y_tr, x_te, y_te):
            import matplotlib.pyplot as plt
            y_tr_p = mod.predict(x_tr)
            y_te_p = mod.predict(x_te)
            y_tr_cal = cal.predict(np.expand_dims(y_tr_p, 1))
            y_te_cal = cal.predict(np.expand_dims(y_te_p, 1))
            plt.close('all')
            fig = plt.figure(constrained_layout=True, figsize=(14, 10))
            gs = fig.add_gridspec(4, 5)
            ax = fig.add_subplot(gs[0, 0])
            histograms(ax, y_tr_p, y_tr, "Train")
            ax = fig.add_subplot(gs[1, 0])
            histograms(ax, y_te_p, y_te, "Test")

            ax = fig.add_subplot(gs[0, 1])
            scatter(ax, y_tr_p, y_tr)
            ax = fig.add_subplot(gs[1, 1])
            scatter(ax, y_te_p, y_te)

            ax = fig.add_subplot(gs[0, 2])
            scatter(ax, y_tr_cal, y_tr)
            ax.set_xlabel("Pred Calibrated")
            ax = fig.add_subplot(gs[1, 2])
            scatter(ax, y_te_cal, y_te)
            ax.set_xlabel("Pred Calibrated")

            ax = fig.add_subplot(gs[2, 0])
            scatter(ax, y_tr, x_tr[:, 0].ravel(), joint_lim=False)
            ax.set_title('Applicability (%s) Train' % ax.get_title())
            ax.set_ylabel("Applicability")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[2, 1])
            scatter(ax, y_tr, x_tr[:, 1].ravel(), joint_lim=False)
            ax.set_title('Robustness (%s) Train' % ax.get_title())
            ax.set_ylabel("Robustness")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[2, 2])
            scatter(ax, y_tr, x_tr[:, 2].ravel(), joint_lim=False)
            ax.set_title('Prior (%s) Train' % ax.get_title())
            ax.set_ylabel("Prior")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[2, 3])
            scatter(ax, y_tr, x_tr[:, 3].ravel(), joint_lim=False)
            ax.set_title('Prior Signature (%s) Train' % ax.get_title())
            ax.set_ylabel("Prior")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[2, 4])
            scatter(ax, y_tr, x_tr[:, 4].ravel(), joint_lim=False)
            ax.set_title('Intensity (%s) Train' % ax.get_title())
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Correlation")

            ax = fig.add_subplot(gs[3, 0])
            scatter(ax, y_te, x_te[:, 0].ravel(), joint_lim=False)
            ax.set_title('Applicability (%s) Test' % ax.get_title())
            ax.set_ylabel("Applicability")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[3, 1])
            scatter(ax, y_te, x_te[:, 1].ravel(), joint_lim=False)
            ax.set_title('Robustness (%s) Test' % ax.get_title())
            ax.set_ylabel("Robustness")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[3, 2])
            scatter(ax, y_te, x_te[:, 2].ravel(), joint_lim=False)
            ax.set_title('Prior (%s) Test' % ax.get_title())
            ax.set_ylabel("Prior")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[3, 3])
            scatter(ax, y_te, x_te[:, 3].ravel(), joint_lim=False)
            ax.set_title('Prior Signature (%s) Test' % ax.get_title())
            ax.set_ylabel("Prior")
            ax.set_xlabel("Correlation")
            ax = fig.add_subplot(gs[3, 4])
            scatter(ax, y_te, x_te[:, 4].ravel(), joint_lim=False)
            ax.set_title('Intensity (%s) Test' % ax.get_title())
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Correlation")

            # ax = fig.add_subplot(gs[1, 3])
            if plots:
                plt.savefig(os.path.join(save_path, 'confidence_stats.png'))
                plt.close()

        def find_p(mod, x_tr, y_tr, x_te, y_te):
            import matplotlib.pyplot as plt
            test = ks_2samp
            ps = []
            ss_te = []
            for p in np.linspace(0, 3, 10):
                w = get_weights(y_tr, p=p)
                mod.fit(x_tr, y_tr, linearregression__sample_weight=w)
                y_te_p = mod.predict(x_te)
                s_te = test(y_te, y_te_p)[0]
                ps += [p]
                ss_te += [s_te]
            p = ps[np.argmin(ss_te)]
            if plots:
                plt.close('all')
                plt.scatter(ps, ss_te)
                plt.title("%.2f" % p)
                plt.savefig(os.path.join(save_path, 'confidence_p.png'))
                plt.close()
            return p

        self.__log.info('Training CONFIDENCE model')

        X, Y = self.save_confidence_distributions(
            siamese, neig_x, train_x, prior_model, prior_sign_model,
            save_path, splits, p_self=p_self)

        # generate train test split
        out_file = os.path.join(save_path, 'data.h5')
        with h5py.File(out_file, "w") as fh:
            for split_name, split_frac, split_idx in splits:
                xs_name = "x_%s" % split_name
                ys_name = "y_%s" % split_name
                self.__log.debug('writing %s: %s' %
                                 (xs_name, str(X[split_idx].shape)))
                fh.create_dataset(xs_name, data=X[split_idx])
                self.__log.debug('writing %s: %s' %
                                 (ys_name, str(Y[split_idx].shape)))
                fh.create_dataset(ys_name, data=Y[split_idx])
        traintest = DataSignature(out_file)
        x_tr = traintest.get_h5_dataset('x_train')
        y_tr = traintest.get_h5_dataset('y_train').ravel()
        x_te = traintest.get_h5_dataset('x_test')
        y_te = traintest.get_h5_dataset('y_test').ravel()

        model = make_pipeline(StandardScaler(), LinearRegression())
        p = find_p(model, x_tr, y_tr, x_te, y_te)
        model.fit(x_te, y_te,
                  linearregression__sample_weight=get_weights(y_te, p=p))
        calibration_model = make_pipeline(StandardScaler(), LinearRegression())
        y_pr = np.expand_dims(model.predict(x_te), 1)
        calibration_model.fit(y_pr, y_te)
        if plots:
            analyze(model, calibration_model, x_tr, y_tr, x_te, y_te)
        model_file = os.path.join(save_path, 'confidence.pkl')
        pickle.dump(model, open(model_file, 'wb'))
        calibration_file = os.path.join(save_path, 'calibration.pkl')
        pickle.dump(calibration_model, open(calibration_file, 'wb'))
        return model, calibration_model

    def save_confidence_distributions(self, siamese, known_x, train_x,
                                      prior_model, prior_sign_model, save_path,
                                      splits, p_self=0.0):
        try:
            from chemicalchecker.tool.siamese import SiameseTriplets
            import faiss
        except ImportError as err:
            raise err

        realistic_fn, trim_mask = self.realistic_subsampling_fn()

        # save neighbors faiss index based on only self train prediction
        self.__log.info('Computing Neighbor Index')
        known_onlyself = mask_keep(self.dataset_idx, known_x)
        known_onlyself_pred = siamese.predict(known_onlyself)
        known_onlyself_neig = faiss.IndexFlatL2(known_onlyself_pred.shape[1])
        known_onlyself_neig.add(known_onlyself_pred.astype(np.float32))
        known_onlyself_neig_file = os.path.join(save_path, 'neig.index')
        faiss.write_index(known_onlyself_neig, known_onlyself_neig_file)
        self.__log.info('Neighbor Index saved: %s' % known_onlyself_neig_file)

        # only self prediction is the ground truth
        unk_onlyself = mask_keep(self.dataset_idx, train_x)
        unk_onlyself_pred = siamese.predict(unk_onlyself)
        unk_notself = mask_exclude(self.dataset_idx, train_x)
        unk_notself_pred = siamese.predict(unk_notself)

        # do applicability domain prediction
        self.__log.info('Computing Applicability Domain')
        applicability, app_range, _ = \
            self.applicability_domain(
                known_onlyself_neig, train_x, siamese, p_self=p_self)
        self.__log.info('Applicability Domain DONE')

        # do conformal prediction (dropout)
        self.__log.info('Computing Conformal Prediction')
        siamese_cp = SiameseTriplets(siamese.model_dir, predict_only=True)
        intensities, robustness, consensus_cp = self.conformal_prediction(
            siamese_cp, train_x, p_self=p_self)
        self.__log.info('Conformal Prediction DONE')

        # predict expected prior
        unk_notself_presence = ~np.isnan(unk_notself[:, ::128])[:, trim_mask]
        prior = prior_model.predict(unk_notself_presence.astype(int))
        prior_sign = prior_sign_model.predict(unk_notself_pred)

        # calculate the error
        log_mse = np.log10(
            np.mean(((unk_onlyself_pred - unk_notself_pred)**2), axis=1))
        log_mse_ad = np.log10(
            np.mean(((unk_onlyself_pred - consensus_cp)**2), axis=1))

        # get correlation between prediction and only self predictions
        correlation = row_wise_correlation(
            unk_onlyself_pred, unk_notself_pred, scaled=True)
        correlation_cp = row_wise_correlation(
            unk_onlyself_pred, consensus_cp, scaled=True)

        # we have all the data to train the confidence model
        self.__log.debug('Saving Confidence Features...')
        conf_features = (
            ('applicability', applicability),
            ('robustness', robustness),
            ('prior', prior),
            ('prior_sign', prior_sign),
            ('intensities', intensities)
        )
        for name, arr in conf_features:
            self.__log.debug('%s %s' % (name, arr.shape))

        features = np.vstack([x[1] for x in conf_features]).T

        know_dist_file = os.path.join(save_path, 'known_dist.h5')
        with h5py.File(know_dist_file, "w") as hf:
            hf.create_dataset('robustness', data=robustness)
            hf.create_dataset('intensity', data=intensities)
            hf.create_dataset('consensus', data=consensus_cp)
            hf.create_dataset('applicability', data=applicability)
            hf.create_dataset('applicability_range', data=app_range)
            hf.create_dataset('prior', data=prior)
            hf.create_dataset('prior_sign', data=prior_sign)
            hf.create_dataset('correlation', data=correlation)
            hf.create_dataset('correlation_consensus', data=correlation_cp)
            hf.create_dataset('log_mse', data=log_mse)
            hf.create_dataset('log_mse_consensus', data=log_mse_ad)

        # save plot
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        variables = ['applicability', 'robustness', 'intensity', 'prior',
                     'prior_sign', 'correlation', 'correlation_consensus_cp']

        def corr(x, y, **kwargs):
            coef = np.corrcoef(x, y)[0][1]
            label = r'$\rho$ = ' + str(round(coef, 2))
            ax = plt.gca()
            ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)

        # Create a pair grid instance
        df = pd.DataFrame(columns=variables)
        df['applicability'] = applicability
        df['robustness'] = robustness
        df['intensity'] = intensities
        df['prior'] = prior
        df['prior_sign'] = prior_sign
        df['correlation'] = correlation
        df['correlation_consensus_cp'] = correlation_cp

        # Map the plots to the locations
        for split_name, split_frac, split_idx in splits:
            grid = sns.PairGrid(data=df.loc[split_idx], vars=variables, size=4)
            grid = grid.map_upper(plt.scatter, color='darkred')
            grid = grid.map_upper(corr)
            grid = grid.map_lower(sns.kdeplot, cmap='Reds')
            grid = grid.map_diag(
                plt.hist, bins=10, edgecolor='k', color='darkred')
            plot_file = os.path.join(
                save_path, 'known_dist_%s.png' % split_name)
            plt.savefig(plot_file)

        return features, correlation

    def applicability_domain(self, neig_index, features, siamese,
                             dropout_fn=None, app_range=None, n_samples=1,
                             p_self=0.0, subsampling=False):
        # applicability is whether not-self preds is close to only-self preds
        # neighbors between 5 and 25 depending on the size of the dataset
        app_thr = int(np.clip(np.log10(self.neig_sign.shape[0])**2, 5, 25))
        if subsampling:
            if dropout_fn is None:
                dropout_fn, _ = self.realistic_subsampling_fn()
            preds, dists, ranges = list(), list(), list()
            for i in range(n_samples):
                pred = siamese.predict(features,
                                       dropout_fn=partial(
                                           dropout_fn, p_self=p_self),
                                       dropout_samples=1)
                pred = np.mean(pred, axis=1)
                only_self_dists, _ = neig_index.search(pred, app_thr)
                if app_range is None:
                    d_min = np.min(only_self_dists)
                    d_max = np.max(only_self_dists)
                else:
                    d_min = app_range[0]
                    d_max = app_range[1]
                curr_app_range = np.array([d_min, d_max])
                preds.append(pred)
                dists.append(only_self_dists)
                ranges.append(curr_app_range)
            consensus = np.mean(np.stack(preds, axis=2), axis=2)
            app_range = [np.min(np.vstack(ranges)[:, 0]),
                         np.max(np.vstack(ranges)[:, 1])]
            # scale and invert distances to get applicability
            apps = list()
            for dist in dists:
                d_min = app_range[0]
                d_max = app_range[1]
                norm_dist = (dist - d_max) / (d_min - d_max)
                applicability = np.mean(norm_dist, axis=1).flatten()
                apps.append(applicability)
            applicability = np.mean(np.vstack(apps), axis=0)
            return applicability, app_range, consensus
        else:
            pred = siamese.predict(mask_exclude(self.dataset_idx, features))
            dists, _ = neig_index.search(pred, app_thr)
            d_min = np.min(dists)
            d_max = np.max(dists)
            app_range = np.array([d_min, d_max])
            norm_dist = (dists - d_max) / (d_min - d_max)
            applicability = np.mean(norm_dist, axis=1).flatten()
            return applicability, app_range, None

    def conformal_prediction(self, siamese_cp, features, dropout_fn=None,
                             nan_pred=None, n_samples=5, p_self=0.0):
        if dropout_fn is None:
            dropout_fn, _ = self.realistic_subsampling_fn()
        # reference prediction (based on no information)
        if nan_pred is None:
            nan_feat = np.full(
                (1, features.shape[1]), np.nan, dtype=np.float32)
            nan_pred = siamese_cp.predict(nan_feat)
        # draw prediction with sub-sampling
        if dropout_fn is None:
            dropout_fn = partial(subsample, dataset_idx=[self.dataset_idx])
        samples = siamese_cp.predict(mask_exclude(self.dataset_idx, features),
                                     dropout_fn=partial(
            dropout_fn, p_self=p_self),
            dropout_samples=n_samples, cp=True)
        # summarize the predictions as consensus
        consensus = np.mean(samples, axis=1)
        # zeros input (no info) as intensity reference
        centered = consensus - nan_pred
        # measure the intensity (mean of absolute comps)
        intensities = np.mean(np.abs(centered), axis=1).flatten()
        # summarize the standard deviation of components
        robustness = 1 - np.mean(np.std(samples, axis=1), axis=1).flatten()
        return intensities, robustness, consensus

    def plot_validations(self, siamese, dataset_idx, traintest_file, chunk_size=10000,
                         limit=1000, dist_limit=1000):

        def no_mask(idxs, x1_data):
            return x1_data

        def read_h5(sign, idxs):
            with h5py.File(sign.data_path, "r") as hf:
                V = hf["x_test"][idxs]
            return V

        def read_unknown(sign, forbidden_idxs, max_n=100000):
            with h5py.File(sign.data_path, "r") as hf:
                V = hf["x_test"][:max_n]
            forbidden_idxs = set(forbidden_idxs)
            unknown_idxs = [i for i in range(
                0, max_n) if i not in forbidden_idxs]
            return V[unknown_idxs]

        import matplotlib.pyplot as plt
        import seaborn as sns
        import itertools
        from MulticoreTSNE import MulticoreTSNE

        mask_fns = {
            'ALL': partial(no_mask, dataset_idx),
            'NOT-SELF': partial(mask_exclude, dataset_idx),
            'ONLY-SELF': partial(mask_keep, dataset_idx),
        }

        all_inchikeys = self.get_universe_inchikeys()
        traintest = DataSignature(traintest_file)
        train_inks = traintest.get_h5_dataset('keys_train')
        test_inks = traintest.get_h5_dataset('keys_test')
        train_idxs = np.argwhere(np.isin(all_inchikeys, train_inks)).flatten()
        test_idxs = np.argwhere(np.isin(all_inchikeys, test_inks)).flatten()
        try:
            unknown_idxs = np.array(
                list(set(np.arange(len(all_inchikeys))) - (set(train_idxs) | set(test_idxs))))
            unknown_idxs = np.sort(np.random.choice(
                unknown_idxs, 5000, replace=False))
        except:
            unknown_idxs = np.array([])

        # predict
        pred = dict()
        pred_file = os.path.join(siamese.model_dir, 'plot_preds.pkl')
        if not os.path.isfile(pred_file):
            self.__log.info('VALIDATION: Predicting train.')
            pred['train'] = dict()
            full_x = DataSignature(self.sign2_universe)
            train = read_h5(full_x, train_idxs[:4000])

            for name, mask_fn in mask_fns.items():
                pred['train'][name] = siamese.predict(mask_fn(train))
            del train
            self.__log.info('VALIDATION: Predicting test.')
            pred['test'] = dict()
            test = read_h5(full_x, test_idxs[:1000])
            for name, mask_fn in mask_fns.items():
                pred['test'][name] = siamese.predict(mask_fn(test))
            del test
            self.__log.info('VALIDATION: Predicting unknown.')
            pred['unknown'] = dict()
            if np.any(unknown_idxs):
                unknown = read_h5(full_x, unknown_idxs[:5000])
                self.__log.info('Number of unknown %s' % len(unknown))
                for name, mask_fn in mask_fns.items():
                    if name == 'ALL':
                        pred['unknown'][name] = siamese.predict(
                            mask_fn(unknown))
                    else:
                        pred['unknown'][name] = []
                del unknown
            else:
                for name, mask_fn in mask_fns.items():
                    pred['unknown'][name] = []
            pickle.dump(pred, open(pred_file, "wb"))
        else:
            pred = pickle.load(open(pred_file, 'rb'))

        # Plot
        self.__log.info('VALIDATION: Plot correlations.')
        fig, axes = plt.subplots(
            1, 3, sharex=True, sharey=False, figsize=(10, 3))
        combos = itertools.combinations(mask_fns, 2)
        for ax, (n1, n2) in zip(axes.flatten(), combos):
            scaled_corrs = row_wise_correlation(
                pred['train'][n1], pred['train'][n2], scaled=True)
            sns.distplot(scaled_corrs, ax=ax, label='Train')
            scaled_corrs = row_wise_correlation(
                pred['test'][n1], pred['test'][n2], scaled=True)
            sns.distplot(scaled_corrs, ax=ax, label='Test')
            ax.legend()
            ax.set_title(label='%s vs. %s' % (n1, n2))
        fname = 'known_unknown_correlations.png'
        plot_file = os.path.join(siamese.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        for metric in ['euclidean', 'cosine']:
            self.__log.info('VALIDATION: Plot %s distances.' % metric)
            fig, axes = plt.subplots(
                1, 3, sharex=True, sharey=True, figsize=(10, 3))
            for ax, name in zip(axes.flatten(), mask_fns):
                ax.set_title(name)
                dist_known = pdist(pred['train'][name][:dist_limit],
                                   metric=metric)
                sns.distplot(dist_known, label='Train', ax=ax)
                dist_known = pdist(pred['test'][name][:dist_limit],
                                   metric=metric)
                sns.distplot(dist_known, label='Test', ax=ax)
                if len(pred['unknown'][name]) > 0:
                    dist_unknown = pdist(pred['unknown'][name][:dist_limit],
                                         metric=metric)
                    sns.distplot(dist_unknown, label='Unknown', ax=ax)
                ax.legend()
            fname = 'known_unknown_dist_%s.png' % metric
            plot_file = os.path.join(siamese.model_dir, fname)
            plt.savefig(plot_file)
            plt.close()

        self.__log.info('VALIDATION: Plot Projections.')
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                 figsize=(10, 10), dpi=200)
        proj_model = MulticoreTSNE(n_components=2)
        proj_limit = min(500, len(pred['test']['ALL']))
        if np.any(pred['unknown']['ALL']):
            proj_train = np.vstack([
                pred['train']['ALL'][:proj_limit],
                pred['test']['ALL'][:proj_limit],
                pred['unknown']['ALL'][:proj_limit],
                pred['test']['ONLY-SELF'][-proj_limit:]
            ])
            proj = proj_model.fit_transform(proj_train)
            dist_tr = proj[:proj_limit]
            dist_ts = proj[proj_limit:(proj_limit * 2)]
            dist_uk = proj[(proj_limit * 2):(proj_limit * 3)]
            dist_os = proj[(proj_limit * 3):]
        else:
            proj_train = np.vstack([
                pred['train']['ALL'][:proj_limit],
                pred['test']['ALL'][:proj_limit],
                pred['test']['ONLY-SELF'][-proj_limit:]
            ])
            proj = proj_model.fit_transform(proj_train)
            dist_tr = proj[:proj_limit]
            dist_ts = proj[proj_limit:(proj_limit * 2)]
            dist_os = proj[(proj_limit * 2):]

        axes[0][0].set_title('Train')
        axes[0][0].scatter(dist_ts[:, 0], dist_ts[:, 1], s=10, color="grey")
        if np.any(pred['unknown']['ALL']):
            axes[0][0].scatter(dist_uk[:, 0], dist_uk[
                               :, 1], s=10, color="grey")
        axes[0][0].scatter(dist_os[:, 0], dist_os[:, 1], s=10, color="grey")
        axes[0][0].scatter(dist_tr[:, 0], dist_tr[:, 1], s=10, color="#1f77b4")

        axes[0][1].set_title('Test')
        axes[0][1].scatter(dist_tr[:, 0], dist_tr[:, 1], s=10, color="grey")
        if np.any(pred['unknown']['ALL']):
            axes[0][1].scatter(dist_uk[:, 0], dist_uk[
                               :, 1], s=10, color="grey")
        axes[0][1].scatter(dist_os[:, 0], dist_os[:, 1], s=10, color="grey")
        axes[0][1].scatter(dist_ts[:, 0], dist_ts[:, 1], s=10, color="#ff7f0e")

        axes[1][0].set_title('Unknown')
        axes[1][0].scatter(dist_tr[:, 0], dist_tr[:, 1], s=10, color="grey")
        axes[1][0].scatter(dist_ts[:, 0], dist_ts[:, 1], s=10, color="grey")
        axes[1][0].scatter(dist_os[:, 0], dist_os[:, 1], s=10, color="grey")
        if np.any(pred['unknown']['ALL']):
            axes[1][0].scatter(dist_uk[:, 0], dist_uk[
                               :, 1], s=10, color="#2ca02c")

        axes[1][1].set_title('ONLY-SELF')
        axes[1][1].scatter(dist_tr[:, 0], dist_tr[:, 1], s=10, color="grey")
        axes[1][1].scatter(dist_ts[:, 0], dist_ts[:, 1], s=10, color="grey")
        if np.any(pred['unknown']['ALL']):
            axes[1][1].scatter(dist_uk[:, 0], dist_uk[
                               :, 1], s=10, color="grey")
        axes[1][1].scatter(dist_os[:, 0], dist_os[:, 1], s=10, color="#d62728")

        fname = 'known_unknown_projection.png'
        plot_file = os.path.join(siamese.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        self.__log.info('VALIDATION: Plot Subsampling.')
        fname = 'known_unknown_sampling.png'
        plot_file = os.path.join(siamese.model_dir, fname)
        plot_subsample(self, plot_file, self.sign2_coverage, traintest_file,
                       ds=self.dataset)

    def save_sign0_matrix(self, sign0, destination, include_confidence=True,
                          chunk_size=1000):
        """Save matrix of signature 0 and confidence values.

        Args:
            sign0(list): Signature 0 to learn from.
            destination(str): Path where to save the matrix (HDF5 file).
            include_confidence(bool): whether to include confidences.
        """
        self.__log.debug('Saving sign0 traintest to: %s' % destination)
        mask = np.isin(list(self.keys), sign0.keys, assume_unique=True)
        # the following work only if sign0 keys is a subset (or ==) of sign3
        assert(np.all(np.isin(list(sign0.keys), self.keys, assume_unique=True)))
        # shapes?
        common_keys = np.count_nonzero(mask)
        x_shape = (common_keys, sign0.shape[1])
        y_shape = (common_keys, self.shape[1])
        if include_confidence:
            y_shape = (common_keys, self.shape[1] + 5)
        with h5py.File(destination, 'w') as hf_out:
            hf_out.create_dataset('x', x_shape, dtype=np.float32)
            hf_out.create_dataset('y', y_shape, dtype=np.float32)
            with h5py.File(self.data_path, 'r') as hf_in:
                out_start = 0
                for i in tqdm(range(0, self.shape[0], chunk_size)):
                    chunk = slice(i, i + chunk_size)
                    labels = hf_in['V'][chunk][mask[chunk]]
                    if include_confidence:
                        stddev = hf_in['stddev_norm'][chunk][mask[chunk]]
                        stddev = np.expand_dims(stddev, 1)
                        intensity = hf_in['intensity_norm'][chunk][mask[chunk]]
                        intensity = np.expand_dims(intensity, 1)
                        prior = hf_in['prior_norm'][chunk][mask[chunk]]
                        prior = np.expand_dims(prior, 1)
                        novelty = hf_in['novelty_norm'][chunk][mask[chunk]]
                        novelty = np.expand_dims(novelty, 1)
                        confidence = hf_in['confidence'][chunk][mask[chunk]]
                        confidence = np.expand_dims(confidence, 1)
                        labels = np.hstack((labels, stddev, intensity,
                                            prior, novelty, confidence))
                    out_size = labels.shape[0]
                    out_chunk = slice(out_start, out_start + out_size)
                    hf_out['y'][out_chunk] = labels
                    del labels
                    hf_out['x'][out_chunk] = sign0[out_chunk]
                    out_start += out_size

    def learn_sign0(self, sign0, suffix=None, evaluate=True):
        """Learn the signature 3 from sign0.

        This method is used twice. First to evaluate the performances of the
        model. Second to train the final model on the full set of data.

        Args:
            sign0(list): Signature 0 object to learn from.
            params(dict): Dictionary with algorithm parameters.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 3 matrix).
            suffix(str): A suffix for the siamese model path (e.g.
                'sign3/models/smiles_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
            include_confidence(bool): whether to include confidences.
        """
        try:
            from chemicalchecker.tool.smilespred import Smilespred
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")
        # get params and set folder
        model_path = os.path.join(self.model_path, 'smiles_%s' % suffix)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        # initialize model and start learning
        shared_keys = sorted(list(sign0.unique_keys & self.unique_keys))
        _, sign0_V = sign0.get_vectors(shared_keys)
        _, sign3_V = self.get_vectors(shared_keys)
        smpred = Smilespred(
            model_dir=model_path, sign0=sign0_V, sign3=sign3_V,
            evaluate=evaluate)
        self.__log.debug('Smiles pred training on %s' % model_path)
        smpred.fit()
        self.smiles_predictor = smpred
        self.__log.debug('model saved to %s' % model_path)
        if evaluate:
            smpred.evaluate()

    def learn_sign0_conf(self, sign0, reuse=True, suffix=None, evaluate=True):
        """Learn the signature 3 applicability from sign0.

        This method is used twice. First to evaluate the performances of the
        model. Second to train the final model on the full set of data.

        Args:
            sign0(list): Signature 0 object to learn from.
            reuse(bool): Whether to reuse intermediate files (e.g. the
                aggregated signature 3 matrix).
            suffix(str): A suffix for the siamese model path (e.g.
                'sign3/models/smiles_<suffix>').
            evaluate(bool): Whether we are performing a train-test split and
                evaluating the performances (N.B. this is required for complete
                confidence scores)
            include_confidence(bool): whether to include confidences.
        """
        try:
            from chemicalchecker.tool.smilespred import ApplicabilityPredictor
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")
        # get params and set folder
        model_path = os.path.join(self.model_path,
                                  'smiles_applicability_%s' % suffix)
        if not os.path.isdir(model_path):
            reuse = False
            os.makedirs(model_path)
        shared_keys = sorted(list(sign0.unique_keys & self.unique_keys))
        _, sign0_V = sign0.get_vectors(shared_keys)
        _, sign3_app_V = self.get_vectors(shared_keys,
                                          dataset_name='confidence')
        sign3_app_V = sign3_app_V.ravel()
        # initialize model and start learning
        apppred = ApplicabilityPredictor(
            model_dir=model_path, sign0=sign0_V,
            applicability=sign3_app_V, evaluate=evaluate)
        self.__log.debug('Applicability pred training on %s' % model_path)
        if not reuse:
            apppred.fit()
        self.smiles_predictor = apppred
        self.__log.debug('model saved to %s' % model_path)
        if evaluate:
            apppred.evaluate()

    def fit_sign0(self, sign0, suffix=None, include_confidence=True,
                  only_confidence=False):
        """Fit signature 3 from Morgan Fingerprint.

        This method is fitting a model that uses Morgan fingerprint as features
        to predict signature 3. In future other featurization approaches can be
        tested.

        Args:
            sign0(str): Path to the MF file.
            include_confidence(bool): Whether to include confidence score in
                regression problem.
            only_confidence(bool): Whether to only train an additional
                regressor exclusively devoted to confidence.
        """

        # check if performance evaluations need to be done
        # Open sign0:
        sign0 = DataSignature(sign0)
        if not only_confidence:
            if suffix is not None:
                self.learn_sign0(sign0,
                                 suffix=suffix,
                                 evaluate=True)
                return False
            else:
                self.learn_sign0(sign0,
                                 suffix='eval',
                                 evaluate=True)

            # check if we have the final trained model
            self.learn_sign0(sign0,
                             suffix='final',
                             evaluate=False)
        if include_confidence:
            if suffix is not None:
                self.learn_sign0_conf(sign0,
                                      suffix=suffix,
                                      evaluate=True)
                return False
            else:

                dest_file = os.path.join(self.model_path,
                                         'smiles_applicability_eval',
                                         'applicabilitypredictor.h5')
                if not os.path.isfile(dest_file):
                    self.learn_sign0_conf(sign0,
                                          suffix='eval',
                                          evaluate=True)

            # check if we have the final trained model
            dest_file = os.path.join(self.model_path,
                                     'smiles_applicability_final',
                                     'applicabilitypredictor.h5')
            if not os.path.isfile(dest_file):
                self.learn_sign0_conf(sign0,
                                      suffix='final',
                                      evaluate=False)

    def get_predict_fn(self, smiles=True, model='smiles_final'):
        try:
            from chemicalchecker.tool.smilespred import Smilespred
            from chemicalchecker.tool.siamese import SiameseTriplets
        except ImportError as err:
            raise err
        model_path = os.path.join(self.model_path, model)
        if smiles:
            model = Smilespred(model_path)
        else:
            model = SiameseTriplets(model_path, predict_only=True)
        return model.predict

    def get_applicability_predict_fn(self, model='smiles_applicability_final'):
        try:
            from chemicalchecker.tool.smilespred import ApplicabilityPredictor
        except ImportError as err:
            raise err
        model_path = os.path.join(self.model_path, model)
        model = ApplicabilityPredictor(model_path)
        return model.predict

    def predict_from_smiles(self, smiles, dest_file, **kwargs):
        return self.predict_from_string(smiles, dest_file, keytype='SMILES',
                                        **kwargs)

    def predict_from_string(self, molecules, dest_file, keytype='SMILES',
                            chunk_size=1000, predict_fn=None,
                            keys=None, components=128, applicability=True):
        """Given molecuel string, generate MFP and predict sign3.

        Args:
            molecules(list): A list of molecules strings.
            dest_file(str): File where to save the predictions.
            keytype(str): Wether to interpret molecules as InChI or SMILES.
        Returns:
            pred_s3(DataSignature): The predicted signatures as DataSignature
                object.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as err:
            raise err
        # input must be a list, otherwise we make it so
        if isinstance(molecules, str):
            molecules = [molecules]
        # convert input molecules to InChI
        inchies = list()
        if keytype.upper() == 'SMILES':
            for smi in molecules:
                if smi == '':
                    smi = 'INVALID SMILES'
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    self.__log.warning(
                        "Cannot get molecule from SMILES: %s." % smi)
                    inchies.append('INVALID SMILES')
                    continue
                inchi = Chem.rdinchi.MolToInchi(mol)[0]
                self.__log.debug('CONVERTED:', smi, inchi)
                inchies.append(inchi)
        elif keytype.upper() == 'INCHI':
            inchies = molecules
        else:
            raise Exception('Keytype not recognized')
        # load NN
        if predict_fn is None:
            predict_fn = self.get_predict_fn()
        if applicability:
            appl_fn = self.get_applicability_predict_fn()
        # we return a simple DataSignature object (basic HDF5 access)
        pred_s3 = DataSignature(dest_file)
        # load novelty model for more accurate novelty scores (slower)
        with h5py.File(dest_file, "w") as results:
            # initialize V (with NaN in case of failing rdkit) and smiles keys
            results.create_dataset('molecules', data=np.array(
                molecules, DataSignature.string_dtype()))
            if keys is not None:
                results.create_dataset('keys', data=np.array(
                    keys, DataSignature.string_dtype()))
            else:
                results.create_dataset('keys', data=np.array(
                    molecules, DataSignature.string_dtype()))
            if applicability:
                results.create_dataset(
                    'applicability', (len(molecules), 1), dtype=np.float32)
            results.create_dataset(
                'V', (len(molecules), components), dtype=np.float32)
            results.create_dataset("shape", data=(len(molecules), components))
            # compute sign0 (i.e. Morgan fingerprint)
            nBits = 2048
            radius = 2
            # predict by chunk
            for i in tqdm(range(0, len(molecules), chunk_size)):
                chunk = slice(i, i + chunk_size)
                sign0s = list()
                failed = list()
                for idx, inchi in enumerate(inchies[chunk]):
                    try:
                        # read molecules
                        inchi = inchi.encode('ascii', 'ignore')
                        mol = Chem.inchi.MolFromInchi(inchi)
                        if mol is None:
                            raise Exception("Cannot get molecule from string.")
                        info = {}
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius, nBits=nBits, bitInfo=info)
                        bin_s0 = [fp.GetBit(i) for i in range(fp.GetNumBits())]
                        calc_s0 = np.array(bin_s0).astype(np.float32)
                    except Exception as err:
                        # in case of failure append a NaN vector
                        self.__log.warn("%s: %s", inchi, str(err))
                        failed.append(idx)
                        calc_s0 = np.full((nBits, ),  np.nan)
                    finally:
                        sign0s.append(calc_s0)
                # stack input signatures and generate predictions
                sign0s = np.vstack(sign0s)
                preds = predict_fn(sign0s)
                # add NaN when SMILES conversion failed
                if failed:
                    preds[np.array(failed)] = np.full(
                        (components, ),  np.nan)
                # save chunk to H5
                results['V'][chunk] = preds[:, :components]
                # also run applicability prediction
                if applicability:
                    apreds = appl_fn(sign0s)
                    if failed:
                        apreds[np.array(failed)] = np.full(
                            (1, ),  np.nan)
                    results['applicability'][chunk] = apreds[:]
        return pred_s3

    def get_universe_inchikeys(self):
        # get sorted universe inchikeys
        if hasattr(self, 'universe_inchikeys'):
            return self.universe_inchikeys
        inchikeys = set()
        for ds, sign in zip(self.src_datasets, self.sign2_list):
            inchikeys.update(sign.unique_keys)
        inchikeys = sorted(list(inchikeys))
        return inchikeys

    def fit(self, sign2_list=None, sign2_self=None, sign1_self=None,
            sign2_universe=None, partial_universe=None,
            sign2_coverage=None, sign0=None,
            model_confidence=True, save_correlations=False,
            predict_novelty=False, update_preds=True,
            chunk_size=1000, suffix=None, **params):
        """Fit signature 3 given a list of signature 2.

        Args:
            sign2_list(list): List of signature 2 objects to learn from.
            sign2_self(sign2): Signature 2 of the current space.
            sign2_universe(str): Path to the union of all signatures 2 for all
                molecules in the CC universe. (800k x 3200)
            sign2_coverage(str): Path to the coverage of all signatures 2 for
                all molecules in the CC universe. (800k x 25)
            sign0(sign): The signature 0 or any direct vector representation
                of the molecule used to train a sign2 independendent sign3
                predictor.
            model_confidence(bool): Whether to model confidence. That is based
                on standard deviation of prediction with dropout.
            save_correlations(bool) Whether to save the correlation (average,
                tertile, max) for the given input dataset (result of the
                evaluation).
            predict_novelty(bool) Whether to predict molecule novelty score.
            update_preds(bool): Whether to write or update the sign3.h5
            normalize_scores(bool): Whether to normalize confidence scores.
            validations(bool): Whether to perform validation.
            chunk_size(int): Chunk size when writing to sign3.h5
        """
        try:
            from chemicalchecker.tool.siamese import SiameseTriplets
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")
        try:
            import faiss
        except ImportError as err:
            raise err
        BaseSignature.fit(self,  **params)

        # define datasets that will be used
        self.update_status("Getting data")
        if sign2_list is None:
            sign2_list = list()
            cc = self.get_cc()
            for ds in cc.datasets_exemplary():
                sign2_list.append(cc.get_signature('sign2', 'full', ds))
        self.sign2_list = sign2_list
        self.src_datasets = [sign.dataset for sign in sign2_list]
        if sign2_self is None:
            sign2_self = self.get_sign('sign2')
        self.sign2_self = sign2_self
        if sign1_self is None:
            sign1_self = self.get_sign('sign1')
        self.neig_sign = sign1_self

        self.sign2_coverage = sign2_coverage
        self.dataset_idx = np.argwhere(
            np.isin(self.src_datasets, self.dataset)).flatten()
        if self.sign2_coverage is None:
            self.sign2_coverage = os.path.join(self.model_path,
                                               'all_sign2_coverage.h5')
        if not os.path.isfile(self.sign2_coverage):
            sign3.save_sign2_coverage(sign2_list, self.sign2_coverage)

        self.__log.debug('Siamese fit %s based on %s', self.dataset,
                         str(self.src_datasets))

        # build input matrix if not provided (should be since is shared)
        self.sign2_universe = sign2_universe
        if self.sign2_universe is None:
            self.sign2_universe = os.path.join(self.model_path, 'all_sign2.h5')
        if partial_universe is not None:
            sign3.complete_sign2_universe(
                sign2_list, sign2_self,
                partial_universe, self.sign2_universe)
        if not os.path.isfile(self.sign2_universe):
            sign3.save_sign2_universe(sign2_list, self.sign2_universe)

        # check if performance evaluations need to be done
        siamese = None
        prior_mdl = None
        prior_sign_mdl = None
        conf_mdl = None

        if suffix is None:
            eval_model_path = os.path.join(self.model_path, 'siamese_eval')
            eval_file = os.path.join(eval_model_path, 'siamesetriplets.h5')
            if not os.path.isfile(eval_file):
                siamese, prior_mdl, prior_sign_mdl, conf_mdl = self.learn_sign2(
                    self.params['sign2'].copy(), suffix='eval', evaluate=True)
        else:
            eval_model_path = os.path.join(self.model_path,
                                           'siamese_%s' % suffix)
            eval_file = os.path.join(eval_model_path, 'siamesetriplets.h5')
            if not os.path.isfile(eval_file):
                siamese, prior_mdl, prior_sign_mdl, conf_mdl = self.learn_sign2(
                    self.params['sign2'].copy(), suffix=suffix, evaluate=True)
            return False

        # check if we have the final trained model
        final_model_path = os.path.join(self.model_path, 'siamese_final')
        final_file = os.path.join(final_model_path, 'siamesetriplets.h5')
        if not os.path.isfile(final_file):
            # get the learning rate from the siamese eval
            siamese_eval = SiameseTriplets(eval_model_path)
            self.params['sign2']['learning_rate'] = siamese_eval.learning_rate
            siamese = self.learn_sign2(
                self.params['sign2'].copy(), suffix='final', evaluate=False)

        # load models if not already available
        if siamese is None:
            siamese = SiameseTriplets(final_model_path, predict_only=True)
        siamese_cp = SiameseTriplets(siamese.model_dir, predict_only=True)

        if model_confidence:
            # part of confidence is the priors
            if prior_mdl is None:
                prior_path = os.path.join(self.model_path, 'prior_eval')
                prior_file = os.path.join(prior_path, 'prior.pkl')
                # if prior model is not there, retrain confidence
                if not os.path.isfile(prior_file):
                    siamese_eval = SiameseTriplets(eval_model_path)
                    sign2_matrix = os.path.join(self.model_path, 'train.h5')
                    self.train_confidence(siamese_eval.traintest_file,
                                          DataSignature(sign2_matrix), 'eval',
                                          siamese_eval)
                prior_mdl = pickle.load(open(prior_file, 'rb'))

            # part of confidence is the priors based on signatures
            if prior_sign_mdl is None:
                prior_sign_path = os.path.join(
                    self.model_path, 'prior_sign_eval')
                prior_sign_file = os.path.join(prior_sign_path, 'prior.pkl')
                prior_sign_mdl = pickle.load(open(prior_sign_file, 'rb'))

            # another part of confidence is the applicability
            confidence_path = os.path.join(self.model_path, 'confidence_eval')
            neig_file = os.path.join(confidence_path, 'neig.index')
            app_neig = faiss.read_index(neig_file)
            known_dist = os.path.join(confidence_path, 'known_dist.h5')
            app_range = DataSignature(known_dist).get_h5_dataset(
                'applicability_range')
            _, trim_mask = self.realistic_subsampling_fn()

            # and finally the linear combination of scores
            if conf_mdl is None:
                confidence_file = os.path.join(
                    confidence_path, 'confidence.pkl')
                calibration_file = os.path.join(
                    confidence_path, 'calibration.pkl')
                conf_mdl = (pickle.load(open(confidence_file, 'rb')),
                            pickle.load(open(calibration_file, 'rb')))

        # get sorted universe inchikeys
        self.universe_inchikeys = self.get_universe_inchikeys()
        tot_inks = len(self.universe_inchikeys)

        # save universe sign3
        if update_preds:
            self.update_status("Predicting universe Sign3")
            with h5py.File(self.data_path, "a") as results:
                # initialize V and keys datasets
                safe_create(results, 'V', (tot_inks, 128), dtype=np.float32)
                safe_create(results, 'keys',
                            data=np.array(self.universe_inchikeys,
                                          DataSignature.string_dtype()))
                # dataset used to generate the signature
                safe_create(results, 'datasets',
                            data=np.array(self.src_datasets,
                                          DataSignature.string_dtype()))
                known_mask = np.isin(list(self.universe_inchikeys),
                                     list(self.sign2_self.keys),
                                     assume_unique=True)
                # save the mask for know/inknown
                safe_create(results, 'known', data=known_mask)
                safe_create(results, 'shape', data=(tot_inks, 128))
                # the actual confidence value will be stored here
                safe_create(results, 'confidence', (tot_inks,),
                            dtype=np.float32)
                if model_confidence:
                    # this is to store robustness
                    safe_create(results, 'robustness',
                                (tot_inks,), dtype=np.float32)
                    # this is to store applicability
                    safe_create(results, 'applicability',
                                (tot_inks,), dtype=np.float32)
                    # this is to store priors
                    safe_create(results, 'prior',
                                (tot_inks,), dtype=np.float32)
                    # this is to store priors based on signature
                    safe_create(results, 'prior_signature',
                                (tot_inks,), dtype=np.float32)
                if predict_novelty:
                    safe_create(results, 'novelty',
                                (tot_inks, ), dtype=np.float32)
                    safe_create(results, 'outlier',
                                (tot_inks, ), dtype=np.float32)

                # predict signature 3 for universe molecules
                with h5py.File(sign2_universe, "r") as features:
                    # reference prediction (based on no information)
                    nan_feat = np.full((1, features['x_test'].shape[1]),
                                       np.nan, dtype=np.float32)
                    nan_pred = siamese.predict(nan_feat)
                    # read input in chunks
                    for idx in tqdm(range(0, tot_inks, chunk_size),
                                    desc='Predicting'):
                        chunk = slice(idx, idx + chunk_size)
                        feat = features['x_test'][chunk]
                        # predict with final model
                        preds = siamese.predict(feat)
                        results['V'][chunk] = preds
                        # skip modelling confidence if not required
                        if not model_confidence:
                            continue
                        # save confidence natural scores
                        # compute prior from coverage
                        cov = ~np.isnan(feat[:, 0::128])
                        prior = prior_mdl.predict(cov[:, trim_mask])
                        results['prior'][chunk] = prior
                        # and from prediction
                        prior_sign = prior_sign_mdl.predict(preds)
                        results['prior_signature'][chunk] = prior_sign
                        # conformal prediction
                        ints, robs, _ = self.conformal_prediction(
                            siamese_cp, feat, nan_pred=nan_pred)
                        results['robustness'][chunk] = robs
                        # distance from known predictions
                        app, centrality, cons = self.applicability_domain(
                            app_neig, feat, siamese, app_range=app_range,
                            n_samples=1)
                        results['applicability'][chunk] = app
                        # and estimate confidence
                        conf_feats = np.vstack(
                            [app, robs, prior, prior_sign, ints]).T
                        conf_estimate = conf_mdl[0].predict(conf_feats)
                        conf_calib = conf_mdl[1].predict(
                            np.expand_dims(conf_estimate, 1))
                        results['confidence'][chunk] = conf_calib
                    # conpute confidence where self is known
                    known_idxs = np.argwhere(known_mask).flatten()
                    # iterate on chunks of knowns
                    for idx in tqdm(range(0, len(known_idxs), 10000),
                                    desc='Computing Confidence'):
                        chunk = slice(idx, idx + 10000)
                        feat = features['x_test'][known_idxs[chunk]]
                        # predict with all features
                        preds_all = siamese.predict(feat)
                        # predict with only-self features
                        feat_onlyself = mask_keep(self.dataset_idx, feat)
                        preds_onlyself = siamese.predict(feat_onlyself)
                        # confidence is correlation ALL vs. ONLY-SELF
                        corrs = row_wise_correlation(
                            preds_onlyself, preds_all, scaled=True)
                        results['confidence'][known_idxs[chunk]] = corrs

        # use semi-supervised anomaly detection algorithm to predict novelty
        if predict_novelty:
            self.update_status("Predicting novelty")
            self.predict_novelty()

        # last step: learn how to get from A1 sign0 to sign3 directly
        # in order to enable SMILES to sign3 predictions
        if sign0 is not None:
            self.update_status("Training SMILES predictor")
            self.fit_sign0(sign0)

        # save reference
        self.save_reference()
        # finalize signature
        BaseSignature.fit_end(self,  **params)

    def predict_novelty(self, retrain=False, update_sign3=True, cpu=4):
        """Model novelty score via LocalOutlierFactor (semi-supervised).

        Args:
            retrain(bool): Drop old model and train again. (default: False)
            update_sign3(bool): Write novelty scores in h5. (default: True)

        """
        novelty_path = os.path.join(self.model_path, 'novelty')
        if not os.path.isdir(novelty_path):
            os.mkdir(novelty_path)
        novelty_model = os.path.join(novelty_path, 'lof.pkl')
        s2_inks = self.sign2_self.keys
        model = None
        if not os.path.isfile(novelty_model) or retrain:
            self.__log.debug('Training novelty score predictor')
            # fit on molecules available in sign2
            _, predicted = self.get_vectors(s2_inks, dataset_name='V')
            t0 = time()
            model = LocalOutlierFactor(
                novelty=True, metric='euclidean', n_jobs=cpu)
            model.fit(predicted)
            delta = time() - t0
            self.__log.debug('Training took: %s' % delta)
            # serialize for later
            pickle.dump(model, open(novelty_model, 'wb'))
        if update_sign3:
            self.__log.debug('Updating novelty scores')
            if model is None:
                model = pickle.load(open(novelty_model, 'rb'))
            # get scores for known molecules and pair with indexes
            s2_idxs = np.argwhere(np.isin(list(self.keys), s2_inks,
                                          assume_unique=True))
            s2_novelty = model.negative_outlier_factor_
            s2_outlier = [0] * s2_novelty.shape[0]
            assert(s2_idxs.shape[0] == s2_novelty.shape[0])
            # predict scores for other molecules and pair with indexes
            s3_inks = sorted(self.unique_keys - set(s2_inks))
            # no new prediction to add
            if len(s3_inks) == 0:
                return
            s3_idxs = np.argwhere(np.isin(list(self.keys), s3_inks,
                                          assume_unique=True))
            _, s3_pred_sign = self.get_vectors(s3_inks)
            s3_novelty = model.score_samples(s3_pred_sign)
            s3_outlier = model.predict(s3_pred_sign)
            assert(len(s3_inks) == s3_novelty.shape[0])
            ordered_scores = np.array(sorted(
                list(zip(s2_idxs.flatten(), s2_novelty, s2_outlier)) +
                list(zip(s3_idxs.flatten(), s3_novelty, s3_outlier))))
            ordered_novelty = ordered_scores[:, 1]
            ordered_outlier = ordered_scores[:, 2]
            with h5py.File(self.data_path, "r+") as results:
                if 'novelty' in results:
                    del results['novelty']
                results['novelty'] = ordered_novelty
                if 'outlier' in results:
                    del results['outlier']
                results['outlier'] = ordered_outlier

    def predict(self, src_file, dst_file, src_h5_ds='x_test',
                dst_h5_ds='V', model_path=None, chunk_size=1000):
        try:
            from chemicalchecker.tool.siamese import SiameseTriplets
        except ImportError:
            raise ImportError("requires tensorflow " +
                              "https://tensorflow.org")

        if model_path is None:
            model_path = os.path.join(self.model_path, 'siamese_debug')
        self.__log.info('PREDICTING using model from: %s' % model_path)
        self.__log.info('INPUT from: %s' % src_file)
        self.__log.info('OUTPUT goes to: %s' % dst_file)
        siamese = SiameseTriplets(model_path, predict_only=True)
        with h5py.File(src_file, "r") as features:
            with h5py.File(dst_file, "w") as preds:
                # create destination h5 dataset
                tot_inks = features[src_h5_ds].shape[0]
                preds_shape = (tot_inks, 128)
                preds.create_dataset(dst_h5_ds, preds_shape, dtype=np.float32)
                if 'keys' in features:
                    preds.create_dataset('keys', data=features['keys'])
                # predict in chunks
                for idx in tqdm(range(0, tot_inks, chunk_size), desc='PRED'):
                    chunk = slice(idx, idx + chunk_size)
                    feat = features[src_h5_ds][chunk]
                    preds[dst_h5_ds][chunk] = siamese.predict(feat)


def safe_create(h5file, *args, **kwargs):
    if args[0] not in h5file:
        h5file.create_dataset(*args, **kwargs)


def mask_keep(idxs, x1_data):
    # we will fill an array of NaN with values we want to keep
    x1_data_transf = np.zeros_like(x1_data, dtype=np.float32) * np.nan
    for idx in idxs:
        # copy column from original data
        col_slice = slice(idx * 128, (idx + 1) * 128)
        x1_data_transf[:, col_slice] = x1_data[:, col_slice]
    # keep rows containing at least one not-NaN value
    return x1_data_transf


def mask_exclude(idxs, x1_data):
    x1_data_transf = np.copy(x1_data)
    for idx in idxs:
        # set current space to nan
        col_slice = slice(idx * 128, (idx + 1) * 128)
        x1_data_transf[:, col_slice] = np.nan
    # drop rows that only contain NaNs
    return x1_data_transf


def col_wise_correlation(X, Y, scaled=False):
    if scaled:
        X = robust_scale(X)
        Y = robust_scale(Y)
    return row_wise_correlation(X.T, Y.T)


def row_wise_correlation(X, Y, scaled=False):
    if scaled:
        X = robust_scale(X)
        Y = robust_scale(Y)
    var1 = (X.T - np.mean(X, axis=1)).T
    var2 = (Y.T - np.mean(Y, axis=1)).T
    cov = np.mean(var1 * var2, axis=1)
    return cov / (np.std(X, axis=1) * np.std(Y, axis=1))


def linear_pow2(val, slope=0.33, intercept=4.):
    flat_log_size = int(np.floor(val * slope))
    return np.power(2, int(flat_log_size + intercept))


def inverse_linear_pow2(val, slope=100, intercept=4.):
    flat_log_size = int(np.floor(slope / val))
    return np.power(2, int(flat_log_size + intercept))


def layer_size_heuristic(samples, features, clip=(512, 2048)):
    log_size = np.log(samples * features)
    pow2 = linear_pow2(log_size, slope=0.3, intercept=5.)
    return np.clip(pow2, *clip)


def batch_size_heuristic(samples, features, clip=(32, 256)):
    log_size = np.log(samples * features)
    pow2 = linear_pow2(log_size, slope=0.5, intercept=-2.)
    return np.clip(pow2, *clip)


def epoch_per_iteration_heuristic(samples, features, clip=(16, 1024)):
    log_size = np.log(samples * features)
    pow2 = inverse_linear_pow2(log_size, slope=270, intercept=-8)
    return np.clip(pow2, *clip)


def subsampling_probs(sign2_coverage, dataset_idx, trim_threshold=0.1,
                      min_unknown=10000):
    """Extract probabilities for known and unknown of a given dataset."""
    if type(sign2_coverage) == str:
        cov = DataSignature(sign2_coverage).get_h5_dataset('x_test')
    else:
        cov = sign2_coverage
    unknown = cov[(cov[:, dataset_idx] == 0).ravel()]
    known = cov[(cov[:, dataset_idx] == 1).ravel()]
    if unknown.shape[0] < min_unknown:
        unknown = known[:min_unknown]
    # decide which spaces are frequent enought in known (used for trainint)
    trim_mask = (np.sum(known, axis=0) / known.shape[0]) > trim_threshold

    def compute_probs(coverage, max_nr=25):
        # how many dataset per molecule?
        nrs, freq_nrs = np.unique(
            np.sum(coverage, axis=1).astype(int), return_counts=True)
        # frequency based probabilities
        p_nrs = freq_nrs / coverage.shape[0]
        # add minimum probability (corner cases where to use 1 or 2 datasets)
        min_p_nr = np.full(max_nr + 1, min(p_nrs), dtype=np.float32)
        for nr, p_nr in zip(nrs, p_nrs):
            min_p_nr[nr] = p_nr
        # but leave out too large nrs
        min_p_nr[max(nrs) + 1:] = 0.0
        min_p_nr[0] = 0.0
        # normalize (sum of probabilities must be one)
        min_p_nr = min_p_nr / np.sum(min_p_nr)
        # print(np.log10(min_p_nr + 1e-10).astype(int))
        # probabilities to keep a dataset?
        p_keep = np.sum(coverage, axis=0) / coverage.shape[0]
        return min_p_nr, p_keep

    p_nr_known, p_keep_known = compute_probs(known[:, trim_mask])
    unknown[:, dataset_idx] = 0
    p_nr_unknown, p_keep_unknown = compute_probs(unknown[:, trim_mask])
    return trim_mask, p_nr_unknown, p_keep_unknown, p_nr_known, p_keep_known


def subsample(tensor, sign_width=128,
              p_nr=(np.array([1 / 25.] * 25), np.array([1 / 25.] * 25)),
              p_keep=(np.array([1 / 25.] * 25), np.array([1 / 25.] * 25)),
              p_only_self=0.0,
              p_self=0.1,
              dataset_idx=[0],
              **kwargs):
    """Function to subsample stacked data."""
    # it is safe to make a local copy of the input matrix
    new_data = np.copy(tensor)
    # we will have a masking matrix at the end
    mask = np.zeros_like(new_data).astype(bool)
    # if new_data.shape[1] % sign_width != 0:
    #    raise Exception('All signature should be of length %i.' % sign_width)
    for idx, row in enumerate(new_data):
        # the following assumes the stacked signature have a fixed width
        presence = ~np.isnan(row[0::sign_width])
        # case where we show only the space itself
        if np.random.rand() < p_only_self:
            presence_add = np.full_like(presence, False)
            presence_add[dataset_idx] = True
            mask[idx] = np.repeat(presence_add, sign_width)
            continue
        # decide to use probabilities from known or unknown
        if np.random.rand() < p_self:
            p_nr_curr = p_nr[0]
            p_keep_curr = p_keep[0]
        else:
            p_nr_curr = p_nr[1]
            p_keep_curr = p_keep[1]
        # datasets that I can select
        present_idxs = np.argwhere(presence).flatten()
        # how many dataset at most?
        max_add = present_idxs.shape[0]
        # normalize nr dataset probabilities
        p_nr_row = p_nr_curr[1:max_add] / np.sum(p_nr_curr[1:max_add])
        # how many dataset are we keeping?
        try:
            nr_keep = np.random.choice(np.arange(1, len(p_nr_row) + 1),
                                       p=p_nr_row)
        except Exception:
            nr_keep = 1
        # normalize dataset keep probabilities
        p_keep_row = (p_keep_curr[presence] + 1e-10) / \
            (np.sum(p_keep_curr[presence]) + 1e-10)
        nr_keep = np.min([nr_keep, np.sum(p_keep_row > 0)])
        # which ones?
        to_add = np.random.choice(
            present_idxs, nr_keep, p=p_keep_row, replace=False)
        # dataset mask
        presence_add = np.zeros(presence.shape).astype(bool)
        presence_add[to_add] = True
        # from dataset mask to signature mask
        mask[idx] = np.repeat(presence_add, sign_width)
    # make masked dataset NaN
    new_data[~mask] = np.nan
    return new_data


def plot_subsample(sign, plot_file, sign2_coverage, traintest_file, ds='B1.001',
                   p_self=.1, p_only_self=0., limit=10000, max_ds=25):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from chemicalchecker import ChemicalChecker

    cc = ChemicalChecker()

    # get triplet generator
    dataset_idx = np.argwhere(
        np.isin(list(cc.datasets_exemplary()), ds)).flatten()
    trim_mask, p_nr_unknown, p_keep_unknown, p_nr_known, p_keep_known = \
        subsampling_probs(sign2_coverage, dataset_idx)
    trim_dataset_idx = np.argwhere(np.arange(len(trim_mask))[
        trim_mask] == dataset_idx).ravel()[0]
    augment_kwargs = {
        'p_nr': (p_nr_unknown, p_nr_known),
        'p_keep': (p_keep_unknown, p_keep_known),
        'dataset_idx': [trim_dataset_idx],
        'p_only_self': 0.0}
    realistic_fn, trim_mask = sign.realistic_subsampling_fn()
    tr_shape_type_gen = NeighborTripletTraintest.generator_fn(
        traintest_file,
        'train_train',
        batch_size=10,
        replace_nan=np.nan,
        augment_fn=realistic_fn,
        augment_kwargs={'dataset_idx': [trim_dataset_idx]},
        trim_mask=trim_mask,
        train=True,
        standard=False)
    tr_gen = tr_shape_type_gen[2]

    # get known unknown
    cov = DataSignature(sign2_coverage).get_h5_dataset('x_test')
    unknown = cov[(cov[:, dataset_idx] == 0).flatten()]
    known = cov[(cov[:, dataset_idx] == 1).flatten()]

    # get dataset probabilities
    probs_ds = {
        'space': np.array([d[:2] for d in cc.datasets_exemplary()])[trim_mask],
        'p_keep_known': p_keep_known,
        'p_keep_unknown': p_keep_unknown}
    df_probs_ds = pd.DataFrame(probs_ds)
    df_probs_ds = df_probs_ds.melt(id_vars=['space'])
    df_probs_ds['probabilities'] = df_probs_ds['value']

    # get nr probabilities
    nnrs, freq_nrs = np.unique(
        np.sum(unknown, axis=1).astype(int), return_counts=True)
    unknown_nr = np.zeros((max_ds + 1,))
    unknown_nr[nnrs] = freq_nrs
    nnrs, freq_nrs = np.unique(
        np.sum(known, axis=1).astype(int), return_counts=True)
    known_nr = np.zeros((max_ds + 1,))
    known_nr[nnrs] = freq_nrs
    probs_nr = {
        'nr_ds': np.arange(max_ds + 1),
        'p_nr_known': p_nr_known,
        'p_nr_unknown': p_nr_unknown}  # == p_nr
    df_probs_nr = pd.DataFrame(probs_nr)
    df_probs_nr = df_probs_nr.melt(id_vars=['nr_ds'])
    df_probs_nr['probabilities'] = df_probs_nr['value']

    # get sampled dataset presence counts
    ds_a = np.zeros((max_ds,))[trim_mask]
    ds_p = np.zeros((max_ds,))[trim_mask]
    ds_n = np.zeros((max_ds,))[trim_mask]
    ds_o = np.zeros((max_ds,))[trim_mask]
    ds_ns = np.zeros((max_ds,))[trim_mask]
    batch = 0
    for (a, p, n, o, ns), y in tr_gen():
        ds_a += np.sum(~np.isnan(a[:, ::128]), axis=0)
        ds_p += np.sum(~np.isnan(p[:, ::128]), axis=0)
        ds_n += np.sum(~np.isnan(n[:, ::128]), axis=0)
        ds_o += np.sum(~np.isnan(o[:, ::128]), axis=0)
        ds_ns += np.sum(~np.isnan(ns[:, ::128]), axis=0)
        batch += 1
        if batch == 1000:
            break
    trimmed_ds = np.array(list(cc.datasets_exemplary()))[trim_mask]
    sampled_ds = {
        'space': np.array([d[:2] for d in trimmed_ds]),
        'anchor': ds_a,
        'positive': ds_p,
        'negative': ds_n,
        'only-self': ds_o,
        'not-self': ds_ns}
    df_sampled_ds = pd.DataFrame(sampled_ds)
    df_sampled_ds = df_sampled_ds.melt(id_vars=['space'])
    df_sampled_ds['sampled'] = df_sampled_ds['value']

    # get sampled nr dataset
    nr_a = np.zeros((max_ds + 1,))
    nr_p = np.zeros((max_ds + 1,))
    nr_n = np.zeros((max_ds + 1,))
    nr_o = np.zeros((max_ds + 1,))
    nr_ns = np.zeros((max_ds + 1,))
    batch = 0
    for (a, p, n, o, ns), y in tr_gen():
        nr_batch_a = np.sum(~np.isnan(a[:, ::128]), axis=1).astype(int)
        nnrs, freq_nrs = np.unique(nr_batch_a, return_counts=True)
        nr_a[nnrs] += freq_nrs
        nr_batch_p = np.sum(~np.isnan(p[:, ::128]), axis=1).astype(int)
        nnrs, freq_nrs = np.unique(nr_batch_p, return_counts=True)
        nr_p[nnrs] += freq_nrs
        nr_batch_n = np.sum(~np.isnan(n[:, ::128]), axis=1).astype(int)
        nnrs, freq_nrs = np.unique(nr_batch_n, return_counts=True)
        nr_n[nnrs] += freq_nrs
        nr_batch_o = np.sum(~np.isnan(o[:, ::128]), axis=1).astype(int)
        nnrs, freq_nrs = np.unique(nr_batch_o, return_counts=True)
        nr_o[nnrs] += freq_nrs
        nr_batch_ns = np.sum(~np.isnan(ns[:, ::128]), axis=1).astype(int)
        nnrs, freq_nrs = np.unique(nr_batch_ns, return_counts=True)
        nr_ns[nnrs] += freq_nrs
        batch += 1
        if batch == 1000:
            break
    sampled_nr = {
        'nr_ds': np.arange(max_ds + 1),
        'anchor': nr_a,
        'positive': nr_p,
        'negative': nr_n,
        'only-self': nr_o,
        'not-self': nr_ns}
    df_sampled_nr = pd.DataFrame(sampled_nr)
    df_sampled_nr = df_sampled_nr.melt(id_vars=['nr_ds'])
    df_sampled_nr['sampled'] = df_sampled_nr['value']

    # plot

    fig = plt.figure(constrained_layout=True, figsize=(24, 12))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    sns.barplot(x="space", y="probabilities", hue='variable',
                data=df_probs_ds, ax=ax)
    ax = fig.add_subplot(gs[0, 1])
    sns.barplot(x="nr_ds", y="probabilities", hue='variable',
                data=df_probs_nr, ax=ax)
    ax = fig.add_subplot(gs[1, 0])
    sns.barplot(x="space", y="sampled", hue='variable',
                data=df_sampled_ds, ax=ax,
                palette=['gold', 'forestgreen', 'crimson', 'royalblue', 'k'])
    ax = fig.add_subplot(gs[1, 1])
    sns.barplot(x="nr_ds", y="sampled", hue='variable',
                data=df_sampled_nr, ax=ax,
                palette=['gold', 'forestgreen', 'crimson', 'royalblue', 'k'])

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
