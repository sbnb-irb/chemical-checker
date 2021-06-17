import os
import pickle
import scipy.stats as stats

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation

import seaborn as sns
from matplotlib import pyplot as plt

from chemicalchecker.util import logged


@logged
class ApplicabilityPredictor(object):

    def __init__(self, model_dir, sign0=[], applicability=[], evaluate=False):
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

        model = Sequential()
        model.add(Dropout(0.5, input_shape=(2048,)))

        drop = 0.2
        model.add(Dense(2**10))
        model.add(Dropout(drop))
        model.add(Activation('relu'))

        model.add(Dense(2**9))
        model.add(Dropout(drop))
        model.add(Activation('relu'))

        model.add(Dense(2**8))
        model.add(Dropout(drop))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('linear'))

        opt = keras.optimizers.Adam(learning_rate=1e-3)
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
            x=self.sign0, y=self.applicability, epochs=30,
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
        j = sns.jointplot(x[:limit], y[:limit],
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
            j = sns.jointplot(x[:limit], y[:limit],
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
