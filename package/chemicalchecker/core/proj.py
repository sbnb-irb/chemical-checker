"""2D Projections Signature.

Perform and store 2D projections, supporting several manifold/dimensionality
reduction algorithms.
"""
import os
import h5py
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from .projector import PCA
from .projector import UMAP
from .projector import TSNE
from .projector import Default
from .signature_base import BaseSignature
from .signature_data import DataSignature

from chemicalchecker.util import logged
from chemicalchecker.util.plot import Plot


@logged
class proj(BaseSignature, DataSignature):
    """Projection Signature class."""

    def __init__(self, signature_path, dataset, proj_type='Default', **kwargs):
        """Initialize the proj class.

        Args:
            signature_path(str): the path to the signature directory.
            dataset(object): The dataset object with all info related
            kwargs(dict): the key is a projector name with the value being
                a dictionary of its parameters.
        """
        # Calling init on the base class to trigger file existance checks
        BaseSignature.__init__(self, signature_path, dataset, **kwargs)
        self.__log.debug('signature path is: %s', signature_path)

        # define which projector is needed
        self.data_path = os.path.join(signature_path, "proj_%s.h5" % proj_type)
        DataSignature.__init__(self, self.data_path)
        self.__log.debug('data_path: %s', self.data_path)

        self.projector = eval(proj_type)(signature_path, dataset, **kwargs)
        self.proj_type = proj_type
        self.stats_path = self.projector.stats_path
        self.model_path = self.projector.model_path

    def pre_fit_transform(self, signature, n_components=15, batch_size=100):
        """Preprocess the input signature reducing by PCA."""
        preprocess_algo = IncrementalPCA(n_components=n_components,
                                         batch_size=batch_size)
        with h5py.File(signature.data_path, "r") as src:
            src_len = src["V"].shape[0]
            for i in tqdm(range(0, src_len, batch_size), 'PRE fit'):
                chunk = slice(i, i + batch_size)
                preprocess_algo.partial_fit(src["V"][chunk])
        sdtype = DataSignature.string_dtype()
        destination = self.data_path + '_tmp_preprocess'
        pred_proj = DataSignature(destination)
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(destination, "w") as dst:
            dst.create_dataset(
                "keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=np.array(
                ['PCA preprocess'], sdtype))
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=np.array([date_str], sdtype))
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, n_components), dtype='float32')
            for i in tqdm(range(0, src_len, batch_size), 'PRE transform'):
                chunk = slice(i, i + batch_size)
                dst['V'][chunk] = preprocess_algo.transform(src['V'][chunk])
        self.preprocess_algo_path = os.path.join(self.signature_path,
                                                 'preprocess.pkl')
        pickle.dump(preprocess_algo, open(self.preprocess_algo_path, 'wb'))
        return pred_proj

    def pre_predict(self, signature, batch_size=1000):
        """Preprocess the input signature reusing PCA transform."""
        preprocess_algo = pickle.load(open(self.preprocess_algo_path, 'rb'))
        sdtype = DataSignature.string_dtype()
        destination = self.data_path + '_tmp_preprocess'
        pred_proj = DataSignature(destination)
        with h5py.File(signature.data_path, "r") as src, \
                h5py.File(destination, "w") as dst:
            dst.create_dataset(
                "keys", data=src['keys'][:], dtype=sdtype)
            dst.create_dataset("name", data=np.array(
                ['PCA preprocess'], sdtype))
            date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dst.create_dataset("date", data=np.array([date_str], sdtype))
            if 'mappings' in src.keys():
                dst.create_dataset("mappings", data=src['mappings'][:],
                                   dtype=sdtype)
            src_len = src["V"].shape[0]
            dst.create_dataset("V", (src_len, 2), dtype='float32')
            for i in tqdm(range(0, src_len, batch_size), 'transform'):
                chunk = slice(i, i + batch_size)
                dst['V'][chunk] = preprocess_algo.transform(src['V'][chunk])
        return pred_proj

    def fit(self, signature=None, validations=True, preprocess_dims=False,
            batch_size=100, *args, **kwargs):
        """Fit a projection model given a signature."""
        if signature is None:
            signature = self.get_sign(
                'sign' + self.cctype[-1]).get_molset("reference")
        self.__log.info("Input shape: %s" % str(signature.shape))
        if preprocess_dims:
            signature = self.pre_fit_transform(signature,
                                               n_components=preprocess_dims,
                                               batch_size=batch_size)
        self.projector.fit(signature, validations, *args, **kwargs)
        # also predict for full if available
        sign_full = self.get_sign('sign' + self.cctype[-1]).get_molset("full")
        if os.path.isfile(sign_full.data_path):
            self_full = self.get_molset("full")
            self_full = proj(self_full.signature_path,
                             self_full.dataset, proj_type=self.proj_type)
            self.predict(sign_full, self_full.data_path)
            #self.map(self_full.data_path) --> not implemented

    def predict(self, signature, destination, *args, **kwargs):
        """Predict projection for new data."""
        if hasattr(self, 'preprocess_algo_path'):
            signature = self.pre_predict(signature)
        return self.projector.predict(signature, destination, *args, **kwargs)

    def plot(self, kind='shaded', *args, **kwargs):
        """Load projected data and plot it."""
        # load data in memory
        proj_data = self[:]
        # plot projection
        range_x = max(abs(np.min(proj_data[:, 0])),
                      abs(np.max(proj_data[:, 0])))
        range_y = max(abs(np.min(proj_data[:, 1])),
                      abs(np.max(proj_data[:, 1])))
        range_max = max(range_y, range_x)
        frame = range_max / 10.
        range_max += frame
        cmap = kwargs.pop('cmap', 'viridis')
        x_range = kwargs.pop('x_range', (-range_max, range_max))
        y_range = kwargs.pop('y_range', (-range_max, range_max))
        plot_size = kwargs.pop('plot_size', (1000, 1000))
        self.__log.info("Plot %s range x: %s y: %s" % (kind, x_range, y_range))
        plot_path = os.path.join(self.stats_path,
                                 self.projector.__class__.__name__)
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot = Plot(self.dataset, plot_path)
        if kind == 'shaded':
            plot.datashader_projection(
                proj_data,
                cmap=cmap,
                x_range=x_range,
                y_range=y_range,
                plot_size=plot_size,
                **kwargs)
        elif kind == 'largevis':
            plot.projection_plot(proj_data, **kwargs)
        elif kind == 'gaussian_kde':
            plot.projection_gaussian_kde(
                proj_data[:, 0], proj_data[:, 1], **kwargs)

    def plot_over(self, data, name, kind='shaded', **kwargs):
        """Load projected data and plot it."""
        proj_data = self[:]
        # plot projection
        range_x = max(abs(np.min(proj_data[:, 0])),
                      abs(np.max(proj_data[:, 0])))
        range_y = max(abs(np.min(proj_data[:, 1])),
                      abs(np.max(proj_data[:, 1])))
        range_max = max(range_y, range_x)
        frame = range_max / 10.
        range_max += frame
        cmap = kwargs.pop('cmap', 'viridis')
        x_range = (-range_max, range_max)
        y_range = (-range_max, range_max)
        plot_size = kwargs.pop('plot_size', (1000, 1000))
        self.__log.info("Plot range x: %s y: %s" % (x_range, y_range))
        plot_path = os.path.join(self.stats_path,
                                 self.projector.__class__.__name__)
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot = Plot(self.dataset, plot_path)
        if kind == 'shaded':
            plot.datashader_projection(
                data,
                self.projector.__class__.__name__ + '_%s' % name,
                cmap=cmap,
                x_range=x_range,
                y_range=y_range,
                plot_size=plot_size,
                transparent=True,
                **kwargs)
        else:
            raise NotImplementedError

    def plot_category(self, data, name, *args, **kwargs):
        """Load projected data and plot it."""
        # data can be a list of numpy array hence plotting multiple categories
        if isinstance(data, list):
            # stack projetion data
            proj_data = np.vstack(tuple([self[:]] + data))
            # stack categories
            category = np.hstack(tuple(
                [np.ones(x.shape[0]) * i for i, x in enumerate(
                    [self[:]] + data)]))
        else:
            proj_data = np.vstack((self[:], data))
            category = np.hstack(
                (np.ones(self.shape[0]) * 0, np.ones(len(data)) * 1))
        # handle arguments
        range_x = max(abs(np.min(proj_data[:, 0])),
                      abs(np.max(proj_data[:, 0])))
        range_y = max(abs(np.min(proj_data[:, 1])),
                      abs(np.max(proj_data[:, 1])))
        range_max = max(range_y, range_x)
        frame = range_max / 10.
        range_max += frame
        cmap = kwargs.pop('cmap', 'viridis')
        x_range = kwargs.pop('x_range', (-range_max, range_max))
        y_range = kwargs.pop('y_range', (-range_max, range_max))
        plot_size = kwargs.pop('plot_size', (1000, 1000))
        noise_scale = kwargs.pop('noise_scale', None)
        category_colors = kwargs.pop('category_colors', None)
        how = kwargs.pop('how', 'eq_hist')
        transparent = kwargs.pop('transparent', False)
        spread = kwargs.pop('spread', 1)
        self.__log.info("Plot range x: %s y: %s" % (x_range, y_range))
        plot_path = os.path.join(self.stats_path,
                                 self.projector.__class__.__name__)
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot = Plot(self.dataset, plot_path)
        df = plot.datashader_projection(
            proj_data,
            name,
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
            plot_size=plot_size,
            noise_scale=noise_scale,
            category=category,
            category_colors=category_colors,
            how=how,
            transparent=transparent,
            spread=spread,
            **kwargs)
        return df
