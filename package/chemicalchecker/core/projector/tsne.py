import os
import h5py
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from time import time
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression

from chemicalchecker.core.signature_base import BaseSignature
from chemicalchecker.core.signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util.plot import Plot


@logged
class TSNE(BaseSignature, DataSignature):
    """A 2D TSNE."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the projection class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(
            self, signature_path, dataset, **params)
        self.__log.debug('signature path is: %s', signature_path)

        self.proj_name = self.__class__.__name__
        self.data_path = os.path.join(
            signature_path, "proj_%s.h5" % self.proj_name)
        self.model_path = os.path.join(self.model_path, self.proj_name)
        if not os.path.isdir(self.model_path):
            original_umask = os.umask(0)
            os.makedirs(self.model_path, 0o775)
            os.umask(original_umask)
        self.stats_path = os.path.join(self.stats_path, self.proj_name)
        if not os.path.isdir(self.stats_path):
            original_umask = os.umask(0)
            os.makedirs(self.stats_path, 0o775)
            os.umask(original_umask)
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)
        self.name = "_".join([str(self.dataset), "proj", self.proj_name])
        self.oos_mdl_path = os.path.join(self.model_path, 'oos.pkl')

    def fit(self, signature, validations=True, chunk_size=5000,
            oos_predictor=False, proj_params={}, pre_pca=True):
        """Fit to signature data."""
        try:
            from MulticoreTSNE import MulticoreTSNE
        except ImportError:
            raise ImportError("requires MulticoreTSNE " +
                              "http://github.com/DmitryUlyanov/Multicore-TSNE")
        projector = MulticoreTSNE(n_components=2, **proj_params)
        # perform fit
        self.__log.info("Projecting with %s..." % self.__class__.__name__)
        for k, v in proj_params.items():
            self.__log.info('  %s %s', k, v)
        self.__log.info("Input shape: %s" % str(signature.info_h5['V']))
        t_start = time()
        # pre PCA
        if pre_pca:
            # find n_components to get 0.9 explained variance
            ipca = IncrementalPCA(n_components=signature.shape[1])
            with h5py.File(signature.data_path, "r") as src:
                src_len = src["V"].shape[0]
                for i in tqdm(range(0, src_len, chunk_size), 'fit expl_var'):
                    chunk = slice(i, i + chunk_size)
                    ipca.partial_fit(src["V"][chunk])
            nr_comp = np.argmax(ipca.explained_variance_ratio_.cumsum() > 0.9)
            # fit pca
            ipca = IncrementalPCA(n_components=nr_comp)
            with h5py.File(signature.data_path, "r") as src:
                src_len = src["V"].shape[0]
                for i in tqdm(range(0, src_len, chunk_size), 'fit'):
                    chunk = slice(i, i + chunk_size)
                    ipca.partial_fit(src["V"][chunk])
            # transform
            proj_data = list()
            with h5py.File(signature.data_path, "r") as src:
                src_len = src["V"].shape[0]
                for i in tqdm(range(0, src_len, chunk_size), 'transform'):
                    chunk = slice(i, i + chunk_size)
                    proj_data.append(ipca.transform(src["V"][chunk]))
            data = np.vstack(proj_data)
        else:
            # read data
            with h5py.File(signature.data_path, "r") as src:
                data = src["V"][:]
        # do projection
        self.__log.info("Final input shape: %s" % str(data.shape))
        proj_data = projector.fit_transform(data)
        if oos_predictor:
            # tsne does not predict so we train linear model
            mdl = LinearRegression()
            mdl.fit(data, proj_data)
            pickle.dump(mdl, open(self.oos_mdl_path, 'wb'))
        t_end = time()
        t_delta = datetime.timedelta(seconds=t_end - t_start)
        self.__log.info("Projecting took %s" % t_delta)
        # save h5
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(self.data_path, "w") as dst:
            dst.create_dataset(
                "keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=np.array([self.name], sdtype))
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=np.array([date_str], sdtype))
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype='float32')
            for i in tqdm(range(0, src_len, chunk_size), 'write'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = proj_data[chunk]
        # make plot
        plot = Plot(self.dataset, self.stats_path)
        xlim, ylim = plot.projection_plot(proj_data, bw=0.1, levels=10)
        # run validation
        if validations:
            self.validate()
        self.mark_ready()

    def predict(self, signature, destination, chunk_size=100):
        """Predict new projections."""
        if not os.path.isfile(self.oos_mdl_path):
            raise Exception('Out-of-sample predictor was not trained.')
        mdl = pickle.load(open(self.oos_mdl_path, 'rb'))
        # create destination file
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(destination, "w") as dst:
            dst.create_dataset(
                "keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=np.array([self.name], sdtype))
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=np.array([date_str], sdtype))
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype='float32')
            for i in tqdm(range(0, src_len, chunk_size), 'transform'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = mdl.predict(src['V'][chunk])
