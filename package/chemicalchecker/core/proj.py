import os

from .signature_base import BaseSignature
from .signature_data import DataSignature

from .projector import Default
from .projector import PCA

from chemicalchecker.util import logged


@logged
class proj(BaseSignature, DataSignature):
    """A Signature bla bla."""

    def __init__(self, signature_path, dataset, proj_type='Default', **kwargs):
        """Initialize the projection class.

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

    def fit(self, signature, validations=True, *args, **kwargs):
        """Take an input learn a 2D representation."""
        self.projector.fit(signature, validations, *args, **kwargs)

    def predict(self, signature, destination, *args, **kwargs):
        """Predict projection for new data."""
        self.projector.predict(signature, destination, *args, **kwargs)

    def plot(self, kind='shaded', *args, **kwargs):
        """Load projected data and plot it."""
        # load data in memory
        proj_data = self.projector[:]
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
        noise_scale = kwargs.pop('noise_scale', None)
        self.__log.info("Plot %s range x: %s y: %s" % (kind, x_range, y_range))
        plot_path = os.path.join(self.stats_path,
                                 self.projector.__class__.__name__)
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot = Plot(self.dataset, plot_path)
        if kind == 'shaded':
            plot.datashader_projection(
                proj_data,
                self.projector.__class__.__name__,
                cmap=cmap,
                x_range=x_range,
                y_range=y_range,
                plot_size=plot_size,
                noise_scale=noise_scale,
                **kwargs)
        elif kind == 'largevis':
            plot.projection_plot(proj_data, **kwargs)
