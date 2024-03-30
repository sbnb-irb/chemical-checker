import os
import pickle
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
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
class Smilespred(object):

    def __init__(self, model_dir, sign0=[], sign3=[], evaluate=False,
                 save_params=True, **kwargs):
        self.sign0 = sign0
        self.sign3 = sign3[:]
        self.name = self.__class__.__name__.lower()
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
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
        self.layers_sizes = kwargs.get("layers_sizes", [1024, 512, 256, 128])
        self.layers = list()
        # we can pass layers type as strings
        layers = kwargs.get("layers", ['Dense', 'Dense', 'Dense', 'Dense'])
        for l in layers:
            if isinstance(l, str):
                self.layers.append(eval(l))
            else:
                self.layers.append(l)
        self.activations = kwargs.get(
            "activations", ['relu', 'relu', 'relu', 'tanh'])
        self.dropouts = kwargs.get("dropouts", [0.1, 0.1, 0.1, None])

        # save params
        if not os.path.isfile(param_file) and save_params:
            self.__log.debug("Saving parameters to %s" % param_file)
            with open(param_file, "wb") as f:
                pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def build_model(self, load=False):
        def corr(y_true, y_pred):
            x = y_true
            y = y_pred
            mx = K.mean(x, axis=0)
            my = K.mean(y, axis=0)
            xm, ym = x - mx, y - my
            r_num = K.sum(xm * ym)
            x_square_sum = K.sum(xm * xm)
            y_square_sum = K.sum(ym * ym)
            r_den = K.sqrt(x_square_sum * y_square_sum)
            r = r_num / r_den
            return K.mean(r)

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

        # last normalization layer for loss
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=opt, metrics=[corr])
        if load:
            model.load_weights(self.model_file)
        else:
            model.summary()
        self.model = model

    def fit(self, save=True):
        self.build_model()
        self.history = self.model.fit(
            x=self.sign0, y=self.sign3, epochs=self.epochs,
            batch_size=128, validation_split=self.e_split, verbose=2)
        if save:
            self.model.save(self.model_file)
        history_file = os.path.join(
            self.model_dir, "%s_history.pkl" % self.name)
        pickle.dump(self.history.history, open(history_file, 'wb'))
        history_file = os.path.join(self.model_dir, "history.png")
        self._plot_history(history_file)

    def predict(self, lst):
        # Load model
        if self.model is None:
            self.build_model(load=True)
        signs = self.model.predict(lst)
        return signs

    def _plot_history(self, h_file):
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes = axes.flatten()
        self.history.history.setdefault('val_loss', [])
        self.history.history.setdefault('val_corr', [])
        axes[0].set_title('Loss', fontsize=19)
        axes[0].plot(self.history.history['loss'], label='train')
        axes[0].plot(self.history.history['val_loss'], label='test')
        axes[0].set_xlabel('Epochs', fontsize=15)
        axes[0].legend()
        axes[1].set_title('Corr', fontsize=19)
        axes[1].plot(self.history.history['corr'], label='train')
        axes[1].plot(self.history.history['val_corr'], label='test')
        axes[1].set_xlabel('Epochs', fontsize=15)
        axes[1].set_ylim(0.5, 1.0)
        fig.tight_layout()
        plt.savefig(h_file)
        plt.close('all')

    def evaluate(self):
        import faiss
        def sim(a, b):
            return -(cosine(a, b) - 1)
        self.__log.info('Predicting all sign0')
        signp = self.model.predict(self.sign0)

        self.__log.info('VALIDATION: Plot distances.')
        subsample = min(int(len(self.sign0) * self.e_split), 100000)
        p = int(subsample / 2)
        tr_idxs = np.arange(int(len(self.sign0) * self.t_split))
        tr_idxs = np.random.choice(tr_idxs, subsample, replace=False)
        ts_idxs = np.arange(int(len(self.sign0) * self.t_split),
                            len(self.sign0))
        ts_idxs = np.random.choice(ts_idxs, subsample, replace=False)

        tr_e_o = np.linalg.norm(
            self.sign3[tr_idxs[:p]] - self.sign3[tr_idxs[p:]], axis=1)
        tr_e_p = np.linalg.norm(
            signp[tr_idxs[:p]] - signp[tr_idxs[p:]], axis=1)
        tr_s_o = [sim(a, b) for a, b in zip(
            self.sign3[tr_idxs[:p]], self.sign3[tr_idxs[p:]])]
        tr_s_p = [sim(a, b)
                  for a, b in zip(signp[tr_idxs[:p]], signp[tr_idxs[p:]])]
        tr_dif = np.linalg.norm(self.sign3[tr_idxs] - signp[tr_idxs], axis=1)

        ts_e_o = np.linalg.norm(
            self.sign3[ts_idxs[:p]] - self.sign3[ts_idxs[p:]], axis=1)
        ts_e_p = np.linalg.norm(
            signp[ts_idxs[:p]] - signp[ts_idxs[p:]], axis=1)
        ts_s_o = [sim(a, b) for a, b in zip(
            self.sign3[ts_idxs[:p]], self.sign3[ts_idxs[p:]])]
        ts_s_p = [sim(a, b)
                  for a, b in zip(signp[ts_idxs[:p]], signp[ts_idxs[p:]])]
        ts_dif = np.linalg.norm(self.sign3[ts_idxs] - signp[ts_idxs], axis=1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='col')
        axes[0][0].set_title('Train Euclidean distances', fontsize=19)
        sns.histplot(tr_e_o, label='original', ax=axes[0][0])
        sns.histplot(tr_e_p, label='predicted', ax=axes[0][0])
        axes[0][0].legend(fontsize=15)

        axes[0][1].set_title('Train Cosine sims', fontsize=19)
        sns.histplot(tr_s_o, label='original', ax=axes[0][1])
        sns.histplot(tr_s_p, label='predicted', ax=axes[0][1])
        axes[0][1].legend(fontsize=15)

        axes[0][2].set_title('Train Distance original-predicted', fontsize=19)
        sns.histplot(tr_dif, ax=axes[0][2])

        axes[1][0].set_title('Test Euclidean distances', fontsize=19)
        sns.histplot(ts_e_o, label='original', ax=axes[1][0])
        sns.histplot(ts_e_p, label='predicted', ax=axes[1][0])
        axes[1][0].legend(fontsize=15)

        axes[1][1].set_title('Test Cosine sims', fontsize=19)
        sns.histplot(ts_s_o, label='original', ax=axes[1][1])
        sns.histplot(ts_s_p, label='predicted', ax=axes[1][1])
        axes[1][1].legend(fontsize=15)

        axes[1][2].set_title('Test Distance original-predicted', fontsize=19)
        sns.histplot(ts_dif, ax=axes[1][2])

        fname = 'distances.png'
        plot_file = os.path.join(self.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        self.__log.info('VALIDATION: Plot projections.')
        proj_model = MulticoreTSNE(n_components=2, n_jobs=8)
        subs_p = 500
        proj_train = np.vstack([
            self.sign3[tr_idxs][:subs_p],
            signp[tr_idxs][:subs_p],
            self.sign3[ts_idxs][:subs_p],
            signp[tr_idxs][:subs_p]
        ])
        proj = proj_model.fit_transform(proj_train)
        tr_o = proj[:subs_p]
        tr_p = proj[subs_p:subs_p + subs_p]
        ts_o = proj[subs_p + subs_p:subs_p + subs_p + subs_p]
        ts_p = proj[subs_p + subs_p + subs_p:]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0][0].set_title('Tr Original', fontsize=15)
        axes[0][0].scatter(tr_p[:, 0], tr_p[:, 1], s=10, color='grey')
        axes[0][0].scatter(ts_o[:, 0], ts_o[:, 1], s=10, color='grey')
        axes[0][0].scatter(ts_p[:, 0], ts_p[:, 1], s=10, color='grey')
        axes[0][0].scatter(tr_o[:, 0], tr_o[:, 1], s=10, color='#1f77b4')

        axes[0][1].set_title('Tr Predicted', fontsize=15)
        axes[0][1].scatter(tr_o[:, 0], tr_o[:, 1], s=10, color='grey')
        axes[0][1].scatter(ts_o[:, 0], ts_o[:, 1], s=10, color='grey')
        axes[0][1].scatter(ts_p[:, 0], ts_p[:, 1], s=10, color='grey')
        axes[0][1].scatter(tr_p[:, 0], tr_p[:, 1], s=10, color='#ff7f0e')

        axes[1][0].set_title('Ts Original', fontsize=15)
        axes[1][0].scatter(tr_o[:, 0], tr_o[:, 1], s=10, color='grey')
        axes[1][0].scatter(tr_p[:, 0], tr_p[:, 1], s=10, color='grey')
        axes[1][0].scatter(ts_p[:, 0], ts_p[:, 1], s=10, color='grey')
        axes[1][0].scatter(ts_o[:, 0], ts_o[:, 1], s=10, color='#2ca02c')

        axes[1][1].set_title('Ts Predicted', fontsize=15)
        axes[1][1].scatter(tr_o[:, 0], tr_o[:, 1], s=10, color='grey')
        axes[1][1].scatter(tr_p[:, 0], tr_p[:, 1], s=10, color='grey')
        axes[1][1].scatter(ts_o[:, 0], ts_o[:, 1], s=10, color='grey')
        axes[1][1].scatter(ts_p[:, 0], ts_p[:, 1], s=10, color='#d62728')
        fname = 'projections.png'
        plot_file = os.path.join(self.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        self.__log.info('VALIDATION: Plot NN overlap.')
        cpu = os.cpu_count()
        faiss.omp_set_num_threads(cpu)
        subs_nn = 100000
        o_nn = faiss.IndexFlatL2(self.sign3.shape[1])
        o_nn.add( np.array( self.sign3[:subs_nn], dtype='float32') )
        o_n_dist, o_n_idxs = o_nn.search( np.array(self.sign3[:subs_nn], dtype='float32'), 100)

        p_nn = faiss.IndexFlatL2(signp.shape[1])
        p_nn.add( np.array( signp[:subs_nn], dtype='float32') )
        p_n_dist, p_n_idxs = p_nn.search( np.array(signp[:subs_nn], dtype='float32'), 100)

        shared_nn = []
        for i in tqdm(range(len(o_n_idxs))):
            tmp = []
            for num_nn in [5, 20, 50, 100]:
                o_row = set(o_n_idxs[i][:num_nn])
                p_row = set(p_n_idxs[i][:num_nn])
                i_num = len(o_row & p_row)
                tmp.append(i_num)
            shared_nn.append(tmp)
        shared_nn = np.array(shared_nn)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0][0].set_title('5 NN')
        sns.histplot(shared_nn[:, 0], ax=axes[0][0])
        axes[0][0].set_xlabel('Num overlap NN')

        axes[0][1].set_title('20 NN')
        sns.histplot(shared_nn[:, 1], ax=axes[0][1], color='#ff7f0e')
        axes[0][1].set_xlabel('Num overlap NN')

        axes[1][0].set_title('50 NN')
        sns.histplot(shared_nn[:, 2], ax=axes[1][0], color='#2ca02c')
        axes[1][0].set_xlabel('Num overlap NN')

        axes[1][1].set_title('100 NN')
        sns.histplot(shared_nn[:, 3], ax=axes[1][1], color='#d62728')
        axes[1][1].set_xlabel('Num overlap NN')
        fname = 'NN_overlap.png'
        plot_file = os.path.join(self.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        self.__log.info('VALIDATION: Plot NN distances.')
        o_df = []
        for row in tqdm(o_n_dist[:1000]):
            for i, dist in enumerate(row):
                o_df.append([i, dist])
        o_df = np.array(o_df)
        o_df = pd.DataFrame(o_df, columns=['NN', 'dist'])

        p_df = []
        for row in tqdm(p_n_dist[:1000]):
            for i, dist in enumerate(row):
                p_df.append([i, dist])
        p_df = np.array(p_df)
        p_df = pd.DataFrame(p_df, columns=['NN', 'dist'])

        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes.set_xlim(0, 100)
        plt.title('Distances per NN', fontsize=19)
        sns.lineplot(x='NN', y='dist', data=o_df, label='original')
        sns.lineplot(x='NN', y='dist', data=p_df, label='predicted')
        plt.legend()
        plt.xlabel('NN', fontsize=15)
        plt.ylabel('Dist', fontsize=15)
        fname = 'NN_distances.png'
        plot_file = os.path.join(self.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()
