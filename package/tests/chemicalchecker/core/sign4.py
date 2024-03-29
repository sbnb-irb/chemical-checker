"""Signature type 4.

Fixed-length (e.g. 128-d) representation of the data, generalizing signature 3
to unseen molecules. Signatures type 4 are available for any molecule of
interest and have a confidence/applicability measure assigned to them.
"""
import os
import h5py
import numpy as np
from tqdm import tqdm

from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged


@logged
class sign4(BaseSignature, DataSignature):
    """Signature type 4 class."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize a Signature.

        Args:
            signature_path(str): The signature root directory.
            dataset(`Dataset`): `chemicalchecker.database.Dataset` object.
            params(): Parameters, expected keys are:
                * 'sign0_params' for learning based on sign0 
                    (Morgan Fingerprint)
                * 'sign0_conf_params' for learning confidences based on MFP
        """
        # Calling init on the base class to trigger file existence checks
        BaseSignature.__init__(self, signature_path,
                               dataset, **params)
        self.data_path = os.path.join(self.signature_path, 'sign4.h5')
        DataSignature.__init__(self, self.data_path)
        # get parameters or default values
        self.params = dict()
        # parameters to learn from sign0
        default_sign0 = {
            'epochs': 30,
            'cpu': 8,
            'learning_rate': 1e-3,
            'layers': ['Dense', 'Dense', 'Dense', 'Dense'],
            'layers_sizes': [1024, 512, 256, 128],
            'activations': ['relu', 'relu', 'relu', 'tanh'],
            'dropouts': [0.1, 0.1, 0.1, None],
        }
        default_sign0.update(params.get('sign0_params', {}))
        self.params['sign0'] = default_sign0
        # parameters to learn confidence from sign0
        default_sign0_conf = {
            'epochs': 30,
            'cpu': 8,
            'learning_rate': 1e-3,
            'layers': ['Dense', 'Dense', 'Dense', 'Dense'],
            'layers_sizes': [1024, 512, 256, 1],
            'activations': ['relu', 'relu', 'relu', 'linear'],
            'dropouts': [0.5, 0.2, 0.2, None]
        }
        default_sign0_conf.update(params.get('sign0_conf_params', {}))
        self.params['sign0_conf'] = default_sign0_conf
        self._sign0_V = None
        self._sign3_V = None

    @property
    def shared_keys(self):
        return sorted(list(self.sign0.unique_keys & self.sign3.unique_keys))

    @property
    def sign0_vectors(self):
        if self._sign0_V is None:
            self.__log.debug("Reading sign0, this should only be loaded once.")
            _, self._sign0_V = self.sign0.get_vectors(self.shared_keys)
            # make sure the order of features is correct
            if 'features' in self.sign0.info_h5:
                order = np.argsort(
                    self.sign0.get_h5_dataset('features').astype(int))
                self._sign0_V = self._sign0_V[:, order]
        self.__log.debug("sign0 shape: %s" % str(self._sign0_V.shape))
        return self._sign0_V

    @property
    def sign3_vectors(self):
        if self._sign3_V is None:
            self.__log.debug("Reading sign3, this should only be loaded once.")
            _, self._sign3_V = self.sign3.get_vectors(self.shared_keys)
        self.__log.debug("sign3 shape: %s" % str(self._sign3_V.shape))
        return self._sign3_V

    def learn_sign0(self, sign0, sign3, params, suffix=None, evaluate=True):
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
            raise ImportError("requires tensorflow https://tensorflow.org")
        # get params and set folder
        model_path = os.path.join(self.model_path, 'smiles_%s' % suffix)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        # initialize model and start learning
        smpred = Smilespred(
            model_dir=model_path, sign0=self.sign0_vectors,
            sign3=self.sign3_vectors, evaluate=evaluate, **params)
        self.__log.debug('Smiles pred training on %s' % model_path)
        smpred.fit()
        self.smiles_predictor = smpred
        self.__log.debug('model saved to %s' % model_path)
        if evaluate:
            smpred.evaluate()

    def learn_sign0_conf(self, sign0, sign3, params, reuse=True, suffix=None,
                         evaluate=True):
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
            raise ImportError("requires tensorflow https://tensorflow.org")
        # get params and set folder
        model_path = os.path.join(self.model_path,
                                  'smiles_applicability_%s' % suffix)
        if not os.path.isdir(model_path):
            reuse = False
            os.makedirs(model_path)
        _, sign3_app_V = self.sign3.get_vectors(self.shared_keys,
                                                dataset_name='confidence')
        sign3_app_V = sign3_app_V.ravel()
        # initialize model and start learning
        apppred = ApplicabilityPredictor(
            model_dir=model_path, sign0=self.sign0_vectors,
            applicability=sign3_app_V, evaluate=evaluate, **params)
        self.__log.debug('Applicability pred training on %s' % model_path)
        if not reuse:
            apppred.fit()
        self.smiles_predictor = apppred
        self.__log.debug('model saved to %s' % model_path)
        if evaluate:
            apppred.evaluate()

    def fit(self, sign0=None, sign3=None, suffix=None, include_confidence=True,
            only_confidence=False, **kwargs):
        """Fit signature 4 from Morgan Fingerprint.

        This method is fitting a model that uses Morgan fingerprint as features
        to predict signature 3. In future other featurization approaches can be
        tested.

        Args:
            sign0(str): Path to the MFP file (i.e. sign0 of A1.001).
            include_confidence(bool): Whether to include confidence score in
                regression problem.
            only_confidence(bool): Whether to only train an additional
                regressor exclusively devoted to confidence.
        """
        BaseSignature.fit(self, **kwargs)
        # signature specific checks
        if self.molset != "full":
            self.__log.debug("Fit will be done for the full sign4")
            self = self.get_molset("full")
        if sign3 is None:
            sign3 = self.get_sign('sign3').get_molset("full")
        if sign0 is None:
            sign0 = self.get_cc().signature('A1.001', 'sign0')
        if sign0.molset != "full":
            self.__log.debug("Fit will be done using full sign0")
            sign0 = sign0.get_molset("full")
        self.sign0 = sign0
        if sign3.molset != "full":
            self.__log.debug("Fit will be done using full sign3")
            sign3 = sign3.get_molset("full")
        self.sign3 = sign3
        if sign0.shape[0] != sign3.shape[0]:
            self.__log.warning("sign3 and MFP do not have the same nr of "
                               "molecules. This might give bad sign0 recap.")

        # check if performance evaluations need to be done
        if not only_confidence:
            self.update_status("Training SMILES-based signature predictor")
            if suffix is not None:
                self.learn_sign0(sign0, sign3, self.params['sign0'].copy(),
                                 suffix=suffix, evaluate=True)
                return False
            else:
                self.learn_sign0(sign0, sign3, self.params['sign0'].copy(),
                                 suffix='eval', evaluate=True)
            # check if we have the final trained model
            self.update_status("Fitting final SMILES model")
            self.learn_sign0(sign0, sign3, self.params['sign0'].copy(),
                             suffix='final', evaluate=False)
        if include_confidence:
            self.update_status("Training SMILES-based confidence predictor")
            if suffix is not None:
                self.learn_sign0_conf(
                    sign0, sign3, self.params['sign0_conf'].copy(),
                    suffix=suffix, evaluate=True)
                return False
            else:
                dest_file = os.path.join(
                    self.model_path, 'smiles_applicability_eval',
                    'applicabilitypredictor.h5')
                if not os.path.isfile(dest_file):
                    self.learn_sign0_conf(
                        sign0, sign3, self.params['sign0_conf'].copy(),
                        suffix='eval', evaluate=True)
            # check if we have the final trained model
            dest_file = os.path.join(self.model_path,
                                     'smiles_applicability_final',
                                     'applicabilitypredictor.h5')
            if not os.path.isfile(dest_file):
                self.update_status("Fitting final confidence model")
                self.learn_sign0_conf(
                    sign0, sign3, self.params['sign0_conf'].copy(),
                    suffix='final', evaluate=False)
        # predict for CC universe
        self.update_status("Predicting for CC universe")
        self.predict_from_sign0(sign0, self.data_path)
        # save reference
        self.save_reference(overwrite=True)
        # finalize signature
        BaseSignature.fit_end(self, **kwargs)

    def get_predict_fn(self, model='smiles_final'):
        try:
            from chemicalchecker.tool.smilespred import Smilespred
        except ImportError as err:
            raise err
        model_path = os.path.join(self.model_path, model)
        model = Smilespred(model_path, save_params=False)
        return model.predict

    def get_applicability_predict_fn(self, model='smiles_applicability_final'):
        try:
            from chemicalchecker.tool.smilespred import ApplicabilityPredictor
        except ImportError as err:
            raise err
        model_path = os.path.join(self.model_path, model)
        model = ApplicabilityPredictor(model_path, save_params=False)
        return model.predict

    def predict_from_smiles(self, smiles, dest_file, **kwargs):
        return self.predict_from_string(smiles, dest_file, keytype='SMILES',
                                        **kwargs)

    def predict_from_sign0(self, sign0, dest_file, chunk_size=1000, y_order=None,
                           **kwargs):
        # load NN
        predict_fn = self.get_predict_fn()
        appl_fn = self.get_applicability_predict_fn()
        # we return a simple DataSignature object (basic HDF5 access)
        pred_s3 = DataSignature(dest_file)
        # load novelty model for more accurate novelty scores (slower)
        with h5py.File(dest_file, "w") as results:
            results.create_dataset('keys', data=np.array(
                sign0.keys, DataSignature.string_dtype()))
            results.create_dataset(
                'applicability', (len(sign0.keys), 1), dtype=np.float32)
            results.create_dataset(
                'V', (len(sign0.keys), 128), dtype=np.float32)
            results.create_dataset("shape", data=(len(sign0.keys), 128))
            # sign0 reorder
            if y_order is None:
                y_order = np.arange(2048)
            if 'features' in sign0.info_h5:
                y_order = np.argsort(sign0.get_h5_dataset('features').astype(int))
            cs = chunk_size
            for chunk, rows in sign0.chunk_iter('V', cs, axis=0, chunk=True):
                rows = rows[:, y_order]
                preds = predict_fn(rows)
                # save chunk to H5
                results['V'][chunk] = preds[:]
                # also run applicability prediction
                apreds = appl_fn(rows)
                results['applicability'][chunk] = apreds[:]
        return pred_s3

    def predict_from_string(self, molecules, dest_file, keytype='SMILES',
                            chunk_size=1000, predict_fn=None, keys=None,
                            components=128, applicability=True, y_order=None):
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
        # reorder as sign0 A1 or leave it as is
        if y_order is None:
            y_order = np.arange(2048)
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
                self.__log.debug('CONVERTED: %s %s', smi, inchi)
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
                sign0s = np.vstack(sign0s)[:, y_order]
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
