import os
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from scipy import stats
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dropout, Lambda, Dense
from tensorflow.keras.layers import Activation, Masking, BatchNormalization
from tensorflow.keras.layers import GaussianNoise, GaussianDropout
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras import regularizers

from MulticoreTSNE import MulticoreTSNE
from scipy.spatial.distance import cosine

import seaborn as sns
from matplotlib import pyplot as plt

from chemicalchecker.util import logged


@logged
class ApplicabilityPredictor(object):

    def __init__(self, model_dir, sign0=[], applicability=[], evaluate=False,
                 save_params=True, **kwargs):
        self.sign0 = sign0
        self.applicability = applicability[:]
        self.name = self.__class__.__name__.lower()
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.is_fit = os.path.isfile(self.model_file)
        if evaluate:
            self.e_split = 0.2
            self.t_split = 1 - self.e_split
        else:
            self.e_split = 0
            self.t_split = 1
        self.model = None

        # check if parameter file exists
        param_file = os.path.join(model_dir, 'params.pkl')
        if os.path.isfile(param_file):
            with open(param_file, 'rb') as h:
                kwargs = pickle.load(h)
            self.__log.info('Parameters loaded from: %s' % param_file)
        self.epochs = int(kwargs.get("epochs", 10))
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.layers_sizes = kwargs.get("layers_sizes", [1024, 512, 256, 1])
        self.layers = list()
        # we can pass layers type as strings
        layers = kwargs.get("layers", ['Dense', 'Dense', 'Dense', 'Dense'])
        for l in layers:
            if isinstance(l, str):
                self.layers.append(eval(l))
            else:
                self.layers.append(l)
        self.activations = kwargs.get(
            "activations", ['relu', 'relu', 'relu', 'linear'])
        self.dropouts = kwargs.get("dropouts", [0.2, 0.2, 0.2, None])

        # save params
        if not os.path.isfile(param_file) and save_params:
            self.__log.debug("Saving parameters to %s" % param_file)
            with open(param_file, "wb") as f:
                pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def build_model(self, load=False):
        def corr(y_true, y_pred):
            x = y_true
            y = y_pred
            mx = K.mean(x)
            my = K.mean(y)
            xm, ym = x - mx, y - my
            r_num = K.sum(xm * ym)
            x_square_sum = K.sum(xm * xm)
            y_square_sum = K.sum(ym * ym)
            r_den = K.sqrt(x_square_sum * y_square_sum)
            r = r_num / r_den
            r = K.maximum(K.minimum(r, 1.0), -1.0)
            return r

        def add_layer(net, layer, layer_size, activation, dropout,
                      use_bias=True, input_shape=False):
            if input_shape is not None:
                if activation == 'selu':
                    net.add(GaussianDropout(rate=0.1, input_shape=input_shape))
                    net.add(layer(layer_size, use_bias=use_bias,
                                  kernel_initializer='lecun_normal'))
                else:
                    net.add(layer(layer_size, use_bias=use_bias,
                                  input_shape=input_shape))
            else:
                if activation == 'selu':
                    net.add(layer(layer_size, use_bias=use_bias,
                                  kernel_initializer='lecun_normal'))
                else:
                    net.add(layer(layer_size, use_bias=use_bias))
            net.add(Activation(activation))
            if dropout is not None:
                net.add(Dropout(dropout))

        def get_model_arch(input_dim, space_dim=128, num_layers=3):
            if input_dim >= space_dim * (2**num_layers):
                layers = [int(space_dim * 2**i)
                          for i in reversed(range(num_layers))]
            else:
                layers = [max(128, int(input_dim / 2**i))
                          for i in range(1, num_layers + 1)]
            return layers

        # Update layers
        if self.layers_sizes == None:
            self.layers_sizes = get_model_arch(
                2048, num_layers=len(self.layers))

        # each goes to a network with the same architechture
        assert(len(self.layers) == len(self.layers_sizes) ==
               len(self.activations) == len(self.dropouts))
        model = Sequential()
        for i, tple in enumerate(zip(self.layers, self.layers_sizes,
                                     self.activations, self.dropouts)):
            layer, layer_size, activation, dropout = tple
            i_shape = None
            if i == 0:
                i_shape = (2048,)
            if i == (len(self.layers) - 1):
                dropout = None
            add_layer(model, layer, layer_size, activation,
                      dropout, input_shape=i_shape)

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=opt,
                      metrics=[
                          keras.metrics.RootMeanSquaredError(name='rmse'),
                          keras.metrics.MeanAbsoluteError(name="mea"),
                          keras.metrics.MeanSquaredLogarithmicError(
                              name="msle"),
                          keras.metrics.LogCoshError(name="logcosh"),
                          corr
                      ])
        if load:
            model.load_weights(self.model_file)
        else:
            model.summary()
        self.model = model

    def fit(self, save=True):
        history_file = os.path.join(
            self.model_dir, "%s_history.pkl" % self.name)
        if self.is_fit:
            if self.model is None:
                self.build_model(load=True)
            self.history = None
            self.history = pickle.load(open(history_file, 'rb'))
            return
        self.build_model()
        self.history = self.model.fit(
            x=self.sign0, y=self.applicability, epochs=self.epochs,
            batch_size=2**8, validation_split=self.e_split, verbose=2)
        if save:
            self.model.save(self.model_file)
        pickle.dump(self.history.history, open(history_file, 'wb'))
        history_file = os.path.join(self.model_dir, "history.png")
        self._plot_history(history_file)
        self._plot_performance()

    def predict(self, lst):
        # Load model
        if self.model is None:
            self.build_model(load=True)
        signs = self.model.predict(lst)
        return signs

    def _plot_history(self, h_file):
        metrics = ['loss', 'rmse', 'mea', 'msle', 'logcosh', 'corr']
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        axes = axes.flatten()
        for met, ax in zip(metrics, axes.flatten()):
            ax.set_title(met, fontsize=19)
            ax.plot(self.history.history[met], label='train', color='red')
            if 'val_%s' % met in self.history.history:
                ax.plot(self.history.history['val_%s' % met],
                        label='test', color='green')
            ax.set_xlabel('Epochs', fontsize=15)
            ax.set_ylim(-0.02, 0.2)
            if met == 'corr':
                ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        plt.savefig(h_file)
        plt.close('all')

    def _plot_performance(self):
        self.__log.info('Predicting all applicabilities')
        preds = self.model.predict(self.sign0).ravel()
        limit = 10000
        tt_split = int(self.sign0.shape[0] * self.t_split)
        x = self.applicability[:tt_split]
        y = preds[:tt_split]

        bbox = dict(facecolor='white', edgecolor='none', pad=3.0, alpha=.7)
        dfplot = pd.DataFrame()
        dfplot['x'] = x[:limit]
        dfplot['y'] = y[:limit]
        j = sns.jointplot( data=dfplot, x='x', y='y',
                          kind="reg", truncate=False,  color="red")
        metric = stats.pearsonr(x, y)[0]
        j.ax_joint.text(0.05, 0.9, 'pearson r: %.2f' % metric,
                        transform=j.ax_joint.transAxes,
                        bbox=bbox)
        j.ax_joint.set_xlabel('True')
        j.ax_joint.set_ylabel('Pred.')
        j.ax_joint.set_xlim(-0.02, 1.02)
        j.ax_joint.set_ylim(-0.02, 1.02)
        plt.savefig(os.path.join(self.model_dir, "true_vs_pred_train.png"))
        plt.close('all')

        if self.t_split < 1:
            x = self.applicability[tt_split:]
            y = preds[tt_split:]
            dfplot = pd.DataFrame()
            dfplot['x'] = x[:limit]
            dfplot['y'] = y[:limit]
            j = sns.jointplot( data=dfplot, x='x', y='y',
                              kind="reg", truncate=False,  color="green")
            metric = stats.pearsonr(x, y)[0]
            j.ax_joint.text(0.05, 0.9, 'pearson r: %.2f' % metric,
                            transform=j.ax_joint.transAxes,
                            bbox=bbox)
            j.ax_joint.set_xlabel('True')
            j.ax_joint.set_ylabel('Pred.')
            j.ax_joint.set_ylim(-0.02, 1.02)
            j.ax_joint.set_xlim(-0.02, 1.02)
            plt.savefig(os.path.join(self.model_dir, "true_vs_pred_test.png"))
            plt.close('all')

    def evaluate(self):
        pass
