import os
import keras
import pickle
import numpy as np
from time import time
from functools import partial

from keras import backend as K
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Input, Dropout, Lambda, Dense, concatenate, BatchNormalization, Activation
from keras.regularizers import l2

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import NeighborTripletTraintest


@logged
class SiameseTriplets(object):
    """Siamese class.

    This class implements a simple siamese neural network based on Keras that
    allows metric learning.
    """

    def __init__(self, model_dir, traintest_file=None, evaluate=False, **kwargs):
        """Initialize the Siamese class.

        Args:
            model_dir(str): Directorty where models will be stored.
            traintest_file(str): Path to the traintest file.
            evaluate(bool): Whether to run evaluation.
        """
        from chemicalchecker.core.signature_data import DataSignature
        # read parameters
        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = float(kwargs.get("learning_rate", 0.0001))
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.dropout = float(kwargs.get("dropout", 0.2))
        self.suffix = str(kwargs.get("suffix", 'eval'))
        self.split = str(kwargs.get("split", 'train'))
        self.layers = kwargs.get("layers", [128])
        self.augment_fn = kwargs.get("augment_fn", None)
        self.augment_kwargs = kwargs.get("augment_kwargs", None)
        self.augment_scale = int(kwargs.get("augment_scale", 1))
        self.margin = float(kwargs.get("margin", 0.2))
        self.patience = float(kwargs.get("patience", 5))

        # internal variables
        self.name = '%s_%s' % (self.__class__.__name__.lower(), self.suffix)
        self.time = 0
        self.output_dim = None
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.model = None
        self.evaluate = evaluate

        # check output path
        if not os.path.exists(model_dir):
            self.__log.warning("Creating model directory: %s", self.model_dir)
            os.mkdir(self.model_dir)

        # check input path
        self.traintest_file = traintest_file
        if self.traintest_file is not None:
            self.traintest_file = os.path.abspath(traintest_file)
            if not os.path.exists(traintest_file):
                raise Exception('Input data file does not exists!')

            # initialize train generator
            self.sharedx = DataSignature(traintest_file).get_h5_dataset('x')
            tr_shape_type_gen = NeighborTripletTraintest.generator_fn(
                self.traintest_file,
                'train_train',
                batch_size=int(self.batch_size / self.augment_scale),
                replace_nan=self.replace_nan,
                sharedx=self.sharedx,
                augment_fn=self.augment_fn,
                augment_kwargs=self.augment_kwargs,
                augment_scale=self.augment_scale)
            self.tr_shapes = tr_shape_type_gen[0]
            self.tr_gen = tr_shape_type_gen[2]()
            self.steps_per_epoch = np.ceil(
                self.tr_shapes[0][0] / self.batch_size)
            self.output_dim = tr_shape_type_gen[0][1][1]

        # initialize validation/test generator
        if evaluate:
            val_shape_type_gen = NeighborTripletTraintest.generator_fn(
                self.traintest_file,
                'test_test',
                batch_size=self.batch_size,
                replace_nan=self.replace_nan,
                sharedx=self.sharedx,
                shuffle=False)
            self.val_shapes = val_shape_type_gen[0]
            self.val_gen = val_shape_type_gen[2]()
            self.validation_steps = np.ceil(
                self.val_shapes[0][0] / self.batch_size)
        else:
            self.val_shapes = None
            self.val_gen = None
            self.validation_steps = None

        # log parameters
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)
        self.__log.info("{:<22}: {:>12}".format("model_dir", self.model_dir))
        if self.traintest_file is not None:
            self.__log.info("{:<22}: {:>12}".format(
                "traintest_file", self.traintest_file))
            tmp = NeighborTripletTraintest(self.traintest_file, 'train_train')
            self.__log.info("{:<22}: {:>12}".format(
                'train_train', str(tmp.get_ty_shapes())))
            if evaluate:
                tmp = NeighborTripletTraintest(self.traintest_file, 'train_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'train_test', str(tmp.get_ty_shapes())))
                tmp = NeighborTripletTraintest(self.traintest_file, 'test_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'test_test', str(tmp.get_ty_shapes())))
        self.__log.info("{:<22}: {:>12}".format(
            "learning_rate", self.learning_rate))
        self.__log.info("{:<22}: {:>12}".format(
            "epochs", self.epochs))
        self.__log.info("{:<22}: {:>12}".format(
            "output_dim", self.output_dim))
        self.__log.info("{:<22}: {:>12}".format(
            "batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "layers", str(self.layers)))
        self.__log.info("{:<22}: {:>12}".format(
            "dropout", str(self.dropout)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_fn", str(self.augment_fn)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_scale", self.augment_scale))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_kwargs", str(self.augment_kwargs)))
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)


    def build_model(self, input_shape, load=False):
        """Compile Keras model

        input_shape(tuple): X dimensions (only nr feat is needed)
        load(bool): Whether to load the pretrained model.
        """

        def dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)


        def euclidean_distance(x, y):
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))

        # we have two inputs
        input_a = Input(shape=input_shape)
        input_p = Input(shape=input_shape)
        input_n = Input(shape=input_shape)

        # each goes to a network with the same architechture
        basenet = Sequential()
        # first layer
        basenet.add(
            Dense(self.layers[0], activation='relu', input_shape=input_shape, use_bias=False))
        if self.dropout is not None:
            basenet.add(Dropout(self.dropout))
        for layer in self.layers[1:-1]:
            basenet.add(Dense(layer, activation='relu', use_bias=False))
            if self.dropout is not None:
                basenet.add(Dropout(self.dropout))
        basenet.add(
            Dense(self.layers[-1], activation='tanh', use_bias=False))
        basenet.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
        #basenet.add(Activation('sigmoid'))

        #basenet.add(BatchNormalization())

        #basenet.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
        basenet.summary()

        encoded_a = basenet(input_a)
        encoded_p = basenet(input_p)
        encoded_n = basenet(input_n)

        merged_vector = concatenate([encoded_a, encoded_p, encoded_n], axis=-1, name='merged_layer')

        model = Model(inputs=[input_a, input_p, input_n], output=merged_vector)

        # define monitored metrics
        def acct(y_true, y_pred):
            total_lenght = y_pred.shape.as_list()[-1]

            anchor = y_pred[:,0:int(total_lenght*1/3)]
            positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
            negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

            acc = K.cast(euclidean_distance(anchor, positive) < euclidean_distance(anchor, negative), anchor.dtype)

            return K.mean(acc)

        def acce(y_true, y_pred):
            total_lenght = y_pred.shape.as_list()[-1]

            msk = K.cast(K.equal(y_true, 0), 'float32')

            anchor = y_pred[:,0:int(total_lenght*1/3)] * msk
            positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)] * msk
            negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)] * msk

            acc = K.cast(euclidean_distance(anchor, positive) < euclidean_distance(anchor, negative), anchor.dtype)

            return K.mean(acc) * 3

        def accm(y_true, y_pred):
            total_lenght = y_pred.shape.as_list()[-1]

            msk = K.cast(K.equal(y_true, 1), 'float32')

            anchor = y_pred[:,0:int(total_lenght*1/3)] * msk
            positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)] * msk
            negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)] * msk

            acc = K.cast(euclidean_distance(anchor, positive) < euclidean_distance(anchor, negative), anchor.dtype)

            return K.mean(acc) * 3

        def acch(y_true, y_pred):
            total_lenght = y_pred.shape.as_list()[-1]

            msk = K.cast(K.equal(y_true, 2), 'float32')

            anchor = y_pred[:,0:int(total_lenght*1/3)] * msk
            positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)] * msk
            negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)] * msk

            acc = K.cast(euclidean_distance(anchor, positive) < euclidean_distance(anchor, negative), anchor.dtype)

            return K.mean(acc) * 3
        
        metrics = [
            acct,
            acce,
            accm,
            acch
        ]

        def triplet_loss(y_true, y_pred, N = 2, beta=2, epsilon=1e-8):
            total_lenght = y_pred.shape.as_list()[-1]

            anchor = y_pred[:,0:int(total_lenght*1/3)]
            positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
            negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

            loss = 1.0 - K.sigmoid(
                K.sum(anchor * positive, axis=-1, keepdims=True) -
                K.sum(anchor * negative, axis=-1, keepdims=True))

            return K.mean(loss)

        # compile and print summary
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            loss=triplet_loss,
            metrics=metrics)
        model.summary()

        # if pre-trained model is specified, load its weights
        self.model = model
        if load:
            self.model.load_weights(self.model_file)
        # this will be the encoder/transformer
        self.transformer = self.model.layers[-2]

    def fit(self, monitor='val_acct'):
        """Fit the model.

        monitor(str): variable to monitor for early stopping.
        """
        # builf model
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        # prepare callbacks
        callbacks = list()

        def mask_keep(idxs, x1_data, x2_data, x3_data):
            # we will fill an array of NaN with values we want to keep
            x1_data_transf = np.zeros_like(x1_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = x1_data[:, col_slice]
            x2_data_transf = np.zeros_like(x2_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x2_data_transf[:, col_slice] = x2_data[:, col_slice]
            x3_data_transf = np.zeros_like(x3_data, dtype=np.float32) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x3_data_transf[:, col_slice] = x3_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan3 = np.isfinite(x3_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2, not_nan3)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            x3_data_transf = x3_data_transf[not_nan]
            return x1_data_transf, x2_data_transf, x3_data_transf

        def mask_exclude(idxs, x1_data, x2_data, x3_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            x2_data_transf = np.copy(x2_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x2_data_transf[:, col_slice] = np.nan
            x3_data_transf = np.copy(x3_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x3_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan3 = np.isfinite(x3_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2, not_nan3)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            x3_data_transf = x3_data_transf[not_nan]
            return x1_data_transf, x2_data_transf, x3_data_transf

        # additional validation sets
        space_idx = self.augment_kwargs['dataset_idx']
        mask_fns = {
            'ALL': None,
            'NOT-SELF': partial(mask_exclude, space_idx),
            'ONLY-SELF': partial(mask_keep, space_idx),
        }
        validation_sets = list()
        vsets = ['train_test', 'test_test']
        if self.evaluate:
            for split in vsets:
                for set_name, mask_fn in mask_fns.items():
                    name = '_'.join([split, set_name])
                    shapes, dtypes, gen = NeighborTripletTraintest.generator_fn(
                        self.traintest_file, split,
                        batch_size=self.batch_size,
                        replace_nan=self.replace_nan,
                        mask_fn=mask_fn,
                        sharedx=self.sharedx,
                        shuffle=False)
                    validation_sets.append((gen, shapes, name))
            additional_vals = AdditionalValidationSets(
                validation_sets, self.model, batch_size=self.batch_size)
            callbacks.append(additional_vals)

        early_stopping = EarlyStopping(
            monitor=monitor,
            verbose=1,
            patience=self.patience,
            mode='max',
            restore_best_weights=True)
        if monitor or not self.evaluate:
            callbacks.append(early_stopping)

        # call fit and save model
        t0 = time()
        self.history = self.model.fit_generator(
            generator=self.tr_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=self.val_gen,
            validation_steps=self.validation_steps,
            shuffle=True)
        self.time = time() - t0
        self.model.save(self.model_file)
        if self.evaluate:
            self.history.history.update(additional_vals.history)

        # check early stopping
        if early_stopping.stopped_epoch != 0:
            self.last_epoch = early_stopping.stopped_epoch - self.patience
        else:
            self.last_epoch = self.epochs

        # save and plot history
        history_file = os.path.join(
            self.model_dir, "%s_history.pkl" % self.name)
        pickle.dump(self.history.history, open(history_file, 'wb'))
        history_file = os.path.join(self.model_dir, "history.png")
        anchor_file = os.path.join(self.model_dir, "anchor_distr.png")
        self._plot_history(self.history.history, vsets, history_file)
        self._plot_anchor_dist(anchor_file)

    def set_predict_scaler(self, scaler):
        self.scaler = scaler

    def predict(self, input_mat):
        """Do predictions.

        prediction_file(str): Path to input file containing Xs.
        split(str): which split to predict.
        batch_size(int): batch size for prediction.
        """
        # load model if not alredy there
        if self.model is None:
            self.build_model((input_mat.shape[1],), load=True)
        no_nans = np.nan_to_num(input_mat)
        if hasattr(self, 'scaler'):
            scaled = self.scaler.fit_transform(no_nans)
        else:
            scaled = no_nans
        return self.transformer.predict(scaled)

    def _plot_history(self, history, vsets, destination):
        """Plot history.

        history(dict): history result from Keras fit method.
        destination(str): path to output file.
        """
        import matplotlib.pyplot as plt

        metrics = list({k.split('_')[-1] for k in history})

        rows = len(metrics)
        cols = len(vsets)

        plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

        c = 1
        for metric in metrics:
            for vset in vsets:
                plt.subplot(rows, cols, c)
                plt.title(metric.capitalize())
                plt.plot(history[metric], label="Train", lw=2, ls='--')
                plt.plot(history['val_' + metric], label="Val", lw=2, ls='--')
                vset_met = [k for k in history if vset in k and metric in k]
                for valset in vset_met:
                    plt.plot(history[valset], label=valset, lw=2)
                plt.legend()
                c += 1

        plt.tight_layout()

        if destination is not None:
            plt.savefig(destination)
        plt.close('all')

    def _plot_anchor_dist(self, plot_file):
        from scipy.spatial.distance import cosine
        import matplotlib.pyplot as plt
        import seaborn as sns

        def sim(a,b):
            return -(cosine(a,b) - 1)


        val_shape_type_gen = NeighborTripletTraintest.generator_fn(
                self.traintest_file,
                'train_test',
                batch_size=self.batch_size,
                replace_nan=self.replace_nan,
                sharedx=self.sharedx,
                shuffle=False)
        trval_gen = val_shape_type_gen[2]()

        vset_dict = {'train_train': self.tr_gen, 'train_test': trval_gen, 'test_test': self.val_gen}
        
        fig, axes = plt.subplots(3, 4, figsize=(22, 15))
        axes = axes.flatten()
        i = 0
        for vset in vset_dict:
            ax = axes[i]
            i += 1
            anchors = list()
            positives = list()
            negatives = list()
            labels = list()
            for inputs, y in vset_dict[vset]:
                anchors.extend(self.predict(inputs[0]))
                positives.extend(self.predict(inputs[1]))
                negatives.extend(self.predict(inputs[2]))
                labels.extend(y)
                if len(anchors) >= 1000:
                    break
            anchors = np.array(anchors)
            positives = np.array(positives)
            negatives = np.array(negatives)
            labels = np.array(labels)

            ap_dists = np.linalg.norm(anchors-positives, axis=1)
            an_dists = np.linalg.norm(anchors-negatives, axis=1)

            mask_e = labels == 0
            mask_m = labels == 1
            mask_h = labels == 2

            ax.set_title('Euclidean '+vset)
            sns.kdeplot(ap_dists[mask_e], label='pos_e', ax=ax, color='limegreen')
            sns.kdeplot(ap_dists[mask_m], label='pos_m', ax=ax, color='forestgreen')
            sns.kdeplot(ap_dists[mask_h], label='pos_h', ax=ax, color='darkgreen')

            sns.kdeplot(an_dists[mask_e], label='neg_e', ax=ax, color='salmon')
            sns.kdeplot(an_dists[mask_m], label='neg_m', ax=ax, color='red')
            sns.kdeplot(an_dists[mask_h], label='neg_h', ax=ax, color='darkred')

            ax.legend()

            ax = axes[i]
            i += 1

            ax.scatter(ap_dists[mask_e], an_dists[mask_e], label='easy', color='green', s=2)
            ax.scatter(ap_dists[mask_m], an_dists[mask_m], label='medium', color='goldenrod', s=2, alpha=0.7)
            ax.scatter(ap_dists[mask_h], an_dists[mask_h], label='hard', color='red', s=2, alpha=0.7)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            ax.set_xlabel('Euc dis positives')
            ax.set_ylabel('Euc dis negatives')


            ax = axes[i]
            i += 1

            ap_sim = np.array([sim(anchors[i], positives[i]) for i in range(len(anchors))])
            an_sim = np.array([sim(anchors[i], negatives[i]) for i in range(len(anchors))])

            ax.set_title('Cosine '+vset)
            sns.kdeplot(ap_sim[mask_e], label='pos_e', ax=ax, color='limegreen')
            sns.kdeplot(ap_sim[mask_m], label='pos_m', ax=ax, color='forestgreen')
            sns.kdeplot(ap_sim[mask_h], label='pos_h', ax=ax, color='darkgreen')
            plt.xlim(-1,1)

            sns.kdeplot(an_sim[mask_e], label='neg_e', ax=ax, color='salmon')
            sns.kdeplot(an_sim[mask_m], label='neg_m', ax=ax, color='red')
            sns.kdeplot(an_sim[mask_h], label='neg_h', ax=ax, color='darkred')
            plt.xlim(-1,1)
            ax.legend()

            ax = axes[i]
            i += 1

            ax.scatter(ap_sim[mask_e], an_sim[mask_e], label='easy', color='green', s=2)
            ax.scatter(ap_sim[mask_m], an_sim[mask_m], label='medium', color='goldenrod', s=2, alpha=0.7)
            ax.scatter(ap_sim[mask_h], an_sim[mask_h], label='hard', color='red', s=2, alpha=0.7)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            ax.set_xlabel('Cos sim positives')
            ax.set_ylabel('Cos sim negatives')

        plt.savefig(plot_file)
        plt.close()








class AdditionalValidationSets(Callback):

    def __init__(self, validation_sets, model, verbose=1, batch_size=None):
        """
        validation_sets(list): list of 3-tuples (val_data, val_targets,
        val_set_name) or 4-tuples (val_data, val_targets, sample_weights,
        val_set_name).
        verbose(int): verbosity mode, 1 or 0.
        batch_size(int): batch size to be used when evaluating on the
        additional datasets.
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = model

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for val_gen, val_shapes, val_set_name in self.validation_sets:
            results = self.model.evaluate_generator(
                val_gen(),
                steps=np.ceil(val_shapes[0][0] / self.batch_size),
                verbose=self.verbose)

            for i, result in enumerate(results):
                name = '_'.join([val_set_name, self.model.metrics_names[i]])
                self.history.setdefault(name, []).append(result)
