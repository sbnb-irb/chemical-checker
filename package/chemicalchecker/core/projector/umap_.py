import os
import h5py
import joblib
import datetime
import numpy as np
from tqdm import tqdm
from time import time

from chemicalchecker.core.signature_base import BaseSignature
from chemicalchecker.core.signature_data import DataSignature

from chemicalchecker.util.plot import Plot
from chemicalchecker.util import logged


@logged
class UMAP(BaseSignature, DataSignature):
    """A 2D UMAP."""

    def __init__(self, signature_path, dataset, **params):
        """Initialize the projection class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related.
        """
        try:
            import umap
        except ImportError:
            raise ImportError("requires umap " +
                              "https://umap-learn.readthedocs.io/en/latest/")
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
        # get default parameters
        self.params = dict(metric='cosine', init='random')
        self.params.update(params)
        # if already fitted load the model and projetions
        self.algo_path = os.path.join(self.model_path, 'algo.pkl')
        self.algo = umap.UMAP(n_components=2, **self.params)

    def fit(self, signature, validations=True, chunk_size=100):
        """Fit to signature data."""
        # perform fit
        self.__log.info("Projecting with %s..." % self.__class__.__name__)
        t_start = time()
        with h5py.File(signature.data_path, "r") as src:
            proj_data = self.algo.fit_transform(src["V"][:])
        t_end = time()
        t_delta = datetime.timedelta(seconds=t_end - t_start)
        self.__log.info("Projecting took %s" % t_delta)
        # save model
        joblib.dump(self.algo, open(self.algo_path, 'wb'))
        # save h5
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(self.data_path, "w") as dst:
            dst.create_dataset("keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=[self.name], dtype=sdtype)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=[date_str], dtype=sdtype)
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype=np.float32)
            for i in tqdm(range(0, src_len, chunk_size), 'write'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = proj_data[chunk]
        # run validation
        if validations:
            self.validate()
        self.mark_ready()

    def predict(self, signature, destination, chunk_size=100, plot=False, plot_kws=None):
        """Predict new projections."""
        # load pickled projector
        try:
            self.algo = joblib.load(open(self.algo_path))
        except Exception as ex:
            self.__log.warning("Cannot load projector: %s" % str(ex))
        # create destination file
        sdtype = DataSignature.string_dtype()
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(destination, "w") as dst:
            dst.create_dataset("keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=[self.name], dtype=sdtype)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=[date_str], dtype=sdtype)
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype=np.float32)
            for i in tqdm(range(0, src_len, chunk_size), 'transform'):
                chunk = slice(i, i + chunk_size)
                dst['V'][chunk] = self.algo.transform(src['V'][chunk])
        prediction = DataSignature(destination)
        if plot:
            # load data in memory
            proj_data = np.vstack(self[:], prediction[:])
            # plot projection
            range_x = max(abs(np.min(proj_data[:, 0])),
                          abs(np.max(proj_data[:, 0])))
            range_y = max(abs(np.min(proj_data[:, 1])),
                          abs(np.max(proj_data[:, 1])))
            range_max = max(range_y, range_x)
            frame = range_max / 10.
            range_max += frame
            cmap = plot_kws.pop('cmap', 'viridis')
            plot_path = plot_kws.pop('plot_path', './')
            x_range = plot_kws.pop('x_range', (-range_max, range_max))
            y_range = plot_kws.pop('y_range', (-range_max, range_max))
            plot_size = plot_kws.pop('plot_size', (1000, 1000))
            noise_scale = plot_kws.pop('noise_scale', None)
            self.__log.info("Plot prediction range x: %s y: %s" % (x_range,
                                                                   y_range))
            if not os.path.isdir(plot_path):
                os.mkdir(plot_path)
            plot = Plot(self.dataset, plot_path)
            plot.datashader_projection(
                proj_data,
                self.projector.__class__.__name__,
                cmap=cmap,
                x_range=x_range,
                y_range=y_range,
                plot_size=plot_size,
                noise_scale=noise_scale,
                **plot_kws)
