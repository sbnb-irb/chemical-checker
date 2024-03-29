import os
import pickle
import numpy as np
from time import time
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dropout, Lambda, Dense
from tensorflow.keras.layers import Activation, Masking, BatchNormalization
from tensorflow.keras.layers import GaussianNoise, AlphaDropout, GaussianDropout
from tensorflow.keras import regularizers

from chemicalchecker.util import logged
from chemicalchecker.util.splitter import TripletIterator
from .callbacks import CyclicLR, LearningRateFinder

MIN_LR = 1e-8
MAX_LR = 1e-1


class AlphaDropoutCP(keras.layers.AlphaDropout):

    def __init__(self, rate, cp=None, noise_shape=None, seed=None, **kwargs):
        super(AlphaDropoutCP, self).__init__(rate, **kwargs)
        self.cp = cp
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        return self.noise_shape if self.noise_shape else K.shape(inputs)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                alpha_p = -alpha * scale

                kept_idx = K.greater_equal(K.random_uniform(noise_shape,
                                                            seed=seed), rate)
                kept_idx = K.cast(kept_idx, K.floatx())

                # Get affine transformation params
                a = ((1 - rate) * (1 + rate * alpha_p ** 2)) ** -0.5
                b = -a * alpha_p * rate

                # Apply mask
                x = inputs * kept_idx + alpha_p * (1 - kept_idx)

                # Do affine transformation
                return a * x + b

            if self.cp:
                return dropped_inputs()
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs


@logged
class SiameseTriplets(object):
    """Siamese class.

    This class implements a simple siamese neural network based on Keras that
    allows metric learning.
    """

    def __init__(self, model_dir, evaluate=False, predict_only=False,
                 plot=True, save_params=True, generator=None, **kwargs):
        """Initialize the Siamese class.

        Args:
            model_dir(str): Directorty where models will be stored.
            traintest_file(str): Path to the traintest file.
            evaluate(bool): Whether to run evaluation.
        """
        from chemicalchecker.core.signature_data import DataSignature
        # check if parameter file exists
        param_file = os.path.join(model_dir, 'params.pkl')
        if os.path.isfile(param_file):
            with open(param_file, 'rb') as h:
                kwargs = pickle.load(h)
            self.__log.info('Parameters loaded from: %s' % param_file)
        # read parameters
        self.epochs = int(kwargs.get("epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 100))
        self.learning_rate = kwargs.get("learning_rate", 'auto')
        self.replace_nan = float(kwargs.get("replace_nan", 0.0))
        self.split = str(kwargs.get("split", 'train'))
        self.layers_sizes = kwargs.get("layers_sizes", [128])
        self.layers = list()
        # we can pass layers type as strings
        layers = kwargs.get("layers", [Dense])
        for l in layers:
            if isinstance(l, str):
                self.layers.append(eval(l))
            else:
                self.layers.append(l)
        self.activations = kwargs.get("activations",
                                      ['relu'])
        self.dropouts = kwargs.get(
            "dropouts", [None])
        self.augment_fn = kwargs.get("augment_fn", None)
        self.augment_kwargs = kwargs.get("augment_kwargs", {})
        self.loss_func = str(kwargs.get("loss_func", 'only_self_loss'))
        self.margin = float(kwargs.get("margin", 1.0))
        self.alpha = float(kwargs.get("alpha", 1.0))
        self.patience = float(kwargs.get("patience", self.epochs))
        self.traintest_file = kwargs.get("traintest_file", None)
        self.onlyself_notself = kwargs.get("onlyself_notself", False)
        self.trim_mask = kwargs.get("trim_mask", None)
        self.steps_per_epoch = kwargs.get("steps_per_epoch", None)
        self.validation_steps = kwargs.get("validation_steps", None)

        # internal variables
        self.name = self.__class__.__name__.lower()
        self.time = 0
        self.model_dir = os.path.abspath(model_dir)
        self.model_file = os.path.join(self.model_dir, "%s.h5" % self.name)
        self.model = None
        self.evaluate = evaluate
        self.plot = plot

        # check output path
        if not os.path.exists(model_dir):
            self.__log.warning("Creating model directory: %s", self.model_dir)
            os.mkdir(self.model_dir)

        # check input path
        self.sharedx = kwargs.get("sharedx", None)
        self.sharedx_trim = kwargs.get("sharedx_trim", None)
        if self.traintest_file is not None:
            traintest_data = DataSignature(self.traintest_file)
            if not predict_only:
                self.traintest_file = os.path.abspath(self.traintest_file)
                if not os.path.exists(self.traintest_file):
                    raise Exception('Input data file does not exists!')

                # initialize train generator
                if generator is None:
                    if self.sharedx is None:
                        self.__log.info("Reading sign2 universe lookup,"
                                        " this should only be loaded once.")
                        self.sharedx = traintest_data.get_h5_dataset('x')
                        full_trim = np.argwhere(np.repeat(self.trim_mask, 128))
                        self.sharedx_trim = self.sharedx[:, full_trim.ravel()]
                    tr_shape_type_gen = TripletIterator.generator_fn(
                        self.traintest_file,
                        'train_train',
                        batch_size=self.batch_size,
                        replace_nan=self.replace_nan,
                        train=True, 
                        augment_fn=self.augment_fn,
                        augment_kwargs=self.augment_kwargs,
                        trim_mask=self.trim_mask,
                        sharedx=self.sharedx,
                        sharedx_trim=self.sharedx_trim,
                        onlyself_notself=self.onlyself_notself)
                else:
                    tr_shape_type_gen = generator
                self.generator = tr_shape_type_gen
                self.tr_shapes = tr_shape_type_gen[0]
                self.tr_gen = tr_shape_type_gen[2]()
                if self.steps_per_epoch is None:
                    self.steps_per_epoch = np.ceil(
                        self.tr_shapes[0][0] / self.batch_size)

            # load the scaler
            if self.onlyself_notself:
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                if os.path.isfile(scaler_path):
                    self.scaler = pickle.load(open(scaler_path, 'rb'))
                    self.__log.info("Using scaler: %s", scaler_path)
                elif 'scaler' in traintest_data.info_h5:
                    scaler_path_tt = traintest_data.get_h5_dataset('scaler')[0]
                    self.__log.info("Using scaler: %s", scaler_path_tt)
                    self.scaler = pickle.load(open(scaler_path_tt, 'rb'))
                    pickle.dump(self.scaler, open(scaler_path, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    self.__log.warning("No scaler has been loaded")

        # initialize validation/test generator
        if evaluate:
            traintest_data = DataSignature(self.traintest_file)
            if self.sharedx is None:
                self.__log.info("Reading sign2 universe lookup,"
                                " this should only be loaded once.")
                self.sharedx = traintest_data.get_h5_dataset('x')
                full_trim = np.argwhere(np.repeat(self.trim_mask, 128))
                self.sharedx_trim = self.sharedx[:, full_trim.ravel()]
            val_shape_type_gen = TripletIterator.generator_fn(
                self.traintest_file,
                'test_test',
                batch_size=self.batch_size,
                shuffle=False,
                train=False,
                replace_nan=self.replace_nan,
                augment_kwargs=self.augment_kwargs,
                augment_fn=self.augment_fn,
                trim_mask=self.trim_mask,
                sharedx=self.sharedx,
                sharedx_trim=self.sharedx_trim,
                onlyself_notself=self.onlyself_notself)
            self.val_shapes = val_shape_type_gen[0]
            self.val_gen = val_shape_type_gen[2]()
            if self.validation_steps is None:
                self.validation_steps = np.ceil(
                    self.val_shapes[0][0] / self.batch_size)
        else:
            self.val_shapes = None
            self.val_gen = None
            self.validation_steps = None

        # log parameters
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)
        self.__log.info("{:<22}: {:>12}".format("model_dir", self.model_dir))
        if self.traintest_file is not None and not predict_only:
            self.__log.info("{:<22}: {:>12}".format(
                "traintest_file", self.traintest_file))
            tmp = TripletIterator(self.traintest_file, 'train_train')
            self.__log.info("{:<22}: {:>12}".format(
                'train_train', str(tmp.get_ty_shapes())))
            if evaluate:
                tmp = TripletIterator(self.traintest_file, 'train_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'train_test', str(tmp.get_ty_shapes())))
                tmp = TripletIterator(self.traintest_file, 'test_test')
                self.__log.info("{:<22}: {:>12}".format(
                    'test_test', str(tmp.get_ty_shapes())))
        self.__log.info("{:<22}: {:>12}".format(
            "learning_rate", self.learning_rate))
        self.__log.info("{:<22}: {:>12}".format(
            "epochs", self.epochs))
        self.__log.info("{:<22}: {:>12}".format(
            "batch_size", self.batch_size))
        self.__log.info("{:<22}: {:>12}".format(
            "layers", str(self.layers)))
        self.__log.info("{:<22}: {:>12}".format(
            "layers_sizes", str(self.layers_sizes)))
        self.__log.info("{:<22}: {:>12}".format(
            "activations", str(self.activations)))
        self.__log.info("{:<22}: {:>12}".format(
            "dropouts", str(self.dropouts)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_fn", str(self.augment_fn)))
        self.__log.info("{:<22}: {:>12}".format(
            "augment_kwargs", str(self.augment_kwargs)))
        self.__log.info("**** %s Parameters: ***" % self.__class__.__name__)

        if not os.path.isfile(param_file) and save_params:
            self.__log.debug("Saving temporary parameters to %s" % param_file)
            with open(param_file+'.tmp', "wb") as f:
                pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.learning_rate == 'auto':
            self.__log.debug("Searching for optimal learning rates.")
            lr = self.find_lr(kwargs, generator=self.generator)
            self.learning_rate = lr
            kwargs['learning_rate'] = self.learning_rate

        if not os.path.isfile(param_file) and save_params:
            self.__log.debug("Saving parameters to %s" % param_file)
            with open(param_file, "wb") as f:
                pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def build_model(self, input_shape, load=False, cp=None):
        """Compile Keras model

        input_shape(tuple): X dimensions (only nr feat is needed)
        load(bool): Whether to load the pretrained model.
        """
        def get_model_arch(input_dim, space_dim=128, num_layers=3):
            if input_dim >= space_dim * (2**num_layers):
                layers = [int(space_dim * 2**i)
                          for i in reversed(range(num_layers))]
            else:
                layers = [max(128, int(input_dim / 2**i))
                          for i in range(1, num_layers + 1)]
            return layers

        def dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        def euclidean_distance(x, y):
            sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, K.epsilon()))

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
                if activation == 'selu':
                    net.add(AlphaDropoutCP(dropout, cp=cp))
                else:
                    net.add(Dropout(dropout))

        # we have two inputs
        input_a = Input(shape=input_shape)
        input_p = Input(shape=input_shape)
        input_n = Input(shape=input_shape)
        if self.onlyself_notself:
            input_o = Input(shape=input_shape)
            input_s = Input(shape=input_shape)

        # Update layers
        if self.layers_sizes == None:
            self.layers_sizes = get_model_arch(
                input_shape[0], num_layers=len(self.layers))

        # each goes to a network with the same architechture
        assert(len(self.layers) == len(self.layers_sizes) ==
               len(self.activations) == len(self.dropouts))
        basenet = Sequential()
        for i, tple in enumerate(zip(self.layers, self.layers_sizes,
                                     self.activations, self.dropouts)):
            layer, layer_size, activation, dropout = tple
            i_shape = None
            if i == 0:
                i_shape = input_shape
            if i == (len(self.layers) - 1):
                dropout = None
            add_layer(basenet, layer, layer_size, activation,
                      dropout, input_shape=i_shape)

        # last normalization layer for loss
        basenet.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
        basenet.summary()

        encodeds = list()
        encodeds.append(basenet(input_a))
        encodeds.append(basenet(input_p))
        encodeds.append(basenet(input_n))
        if self.onlyself_notself:
            encodeds.append(basenet(input_o))
            encodeds.append(basenet(input_s))
        merged_vector = concatenate(encodeds, axis=-1, name='merged_layer')

        inputs = [input_a, input_p, input_n]
        if self.onlyself_notself:
            inputs.extend([input_o, input_s])
        model = Model(inputs=inputs, outputs=merged_vector)

        def split_array(array, sections):
            length = array.shape.as_list()[-1]
            splitted = list()
            for i in range(sections):
                start = int(length * i / sections)
                end = int(length * (i+1) / sections)
                splitted.append(array[:, start:end])
            return splitted

        if self.onlyself_notself:
            def split_output(y_pred):
                anchor, positive, negative, only, n_self = split_array(y_pred, 5)
                return anchor, positive, negative, only, n_self
        else:
            def split_output(y_pred):
                anchor, positive, negative, = split_array(y_pred, 3)
                only, n_self = None, None
                return anchor, positive, negative, only, n_self

        # define monitored metrics
        def accTot(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            acc = K.cast(euclidean_distance(anchor, positive) <
                         euclidean_distance(anchor, negative), anchor.dtype)
            return K.mean(acc)

        def AccEasy(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            msk = K.cast(K.equal(y_true, 0), 'float32')
            prd = self.batch_size / K.sum(msk)
            acc = K.cast(
                euclidean_distance(anchor * msk, positive * msk) <
                euclidean_distance(anchor * msk, negative * msk), anchor.dtype)
            return K.mean(acc) * prd

        def AccMed(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            msk = K.cast(K.equal(y_true, 1), 'float32')
            prd = self.batch_size / K.sum(msk)
            acc = K.cast(
                euclidean_distance(anchor * msk, positive * msk) <
                euclidean_distance(anchor * msk, negative * msk), anchor.dtype)
            return K.mean(acc) * prd

        def AccHard(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            msk = K.cast(K.equal(y_true, 2), 'float32')
            prd = self.batch_size / K.sum(msk)
            acc = K.cast(
                euclidean_distance(anchor * msk, positive * msk) <
                euclidean_distance(anchor * msk, negative * msk), anchor.dtype)
            return K.mean(acc) * prd

        def pearson_r(y_true, y_pred):
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

        def CorANotself(y_true, y_pred):
            anchor, positive, negative, only_self, not_self = split_output(
                y_pred)
            return pearson_r(anchor, not_self)

        def CorAOnlyself(y_true, y_pred):
            anchor, positive, negative, only_self, not_self = split_output(
                y_pred)
            return pearson_r(anchor, only_self)

        def CorNotselfOnlyself(y_true, y_pred):
            anchor, positive, negative, only_self, not_self = split_output(
                y_pred)
            return pearson_r(not_self, only_self)

        metrics = [accTot]
        if self.onlyself_notself:
            metrics.extend([AccEasy,
                            AccMed,
                            AccHard,
                            CorANotself,
                            CorAOnlyself,
                            CorNotselfOnlyself])

        def tloss(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            pos_dist = K.sum(K.square(anchor - positive), axis=1)
            neg_dist = K.sum(K.square(anchor - negative), axis=1)
            basic_loss = pos_dist - neg_dist + self.margin
            loss = K.maximum(basic_loss, 0.0)
            return loss

        def bayesian_tloss(y_true, y_pred):
            anchor, positive, negative, _, _ = split_output(y_pred)
            loss = 1.0 - K.sigmoid(
                K.sum(anchor * positive, axis=-1, keepdims=True) -
                K.sum(anchor * negative, axis=-1, keepdims=True))
            return K.mean(loss)

        def orthogonal_tloss(y_true, y_pred):
            def global_orthogonal_regularization(y_pred):
                anchor, positive, negative, _, _ = split_output(y_pred)
                neg_dis = K.sum(anchor * negative, axis=1)
                dim = K.int_shape(y_pred)[1]
                gor = K.pow(K.mean(neg_dis), 2) + \
                    K.maximum(K.mean(K.pow(neg_dis, 2)) - 1.0 / dim, 0.0)
                return gor

            gro = global_orthogonal_regularization(y_pred) * self.alpha
            loss = tloss(y_true, y_pred)
            return loss + gro

        def only_self_loss(y_true, y_pred):
            def only_self_regularization(y_pred):
                anchor, positive, negative, only_self, _ = split_output(y_pred)
                pos_dist = K.sum(K.square(anchor - only_self), axis=1)
                neg_dist = K.sum(K.square(anchor - negative), axis=1)
                basic_loss = pos_dist - neg_dist + self.margin
                loss = K.maximum(basic_loss, 0.0)
                neg_dis = K.sum(anchor * negative, axis=1)
                dim = K.int_shape(y_pred)[1]
                gor = K.pow(K.mean(neg_dis), 2) + \
                    K.maximum(K.mean(K.pow(neg_dis, 2)) - 1.0 / dim, 0.0)
                return loss + (gor * self.alpha)

            loss = orthogonal_tloss(y_true, y_pred)
            o_self = only_self_regularization(y_pred)
            return loss + o_self

        def penta_loss(y_true, y_pred):
            def only_self_regularization(y_pred):
                anchor, positive, negative, only_self, not_self = split_output(
                    y_pred)
                pos_dist = K.sum(K.square(anchor - only_self), axis=1)
                neg_dist = K.sum(K.square(anchor - negative), axis=1)
                basic_loss = pos_dist - neg_dist + self.margin
                loss = K.maximum(basic_loss, 0.0)
                neg_dis = K.sum(anchor * negative, axis=1)
                dim = K.int_shape(y_pred)[1]
                gor = K.pow(K.mean(neg_dis), 2) + \
                    K.maximum(K.mean(K.pow(neg_dis, 2)) - 1.0 / dim, 0.0)
                return loss + (gor * self.alpha)

            def not_self_regularization(y_pred):
                anchor, positive, negative, only_self, not_self = split_output(
                    y_pred)
                pos_dist = K.sum(K.square(anchor - not_self), axis=1)
                neg_dist = K.sum(K.square(anchor - negative), axis=1)
                basic_loss = pos_dist - neg_dist + self.margin
                loss = K.maximum(basic_loss, 0.0)
                neg_dis = K.sum(anchor * negative, axis=1)
                dim = K.int_shape(y_pred)[1]
                gor = K.pow(K.mean(neg_dis), 2) + \
                    K.maximum(K.mean(K.pow(neg_dis, 2)) - 1.0 / dim, 0.0)
                return loss + (gor * self.alpha)

            def both_self_regularization(y_pred):
                anchor, positive, negative, only_self, not_self = split_output(
                    y_pred)
                pos_dist = K.sum(K.square(not_self - only_self), axis=1)
                neg_dist = K.sum(K.square(not_self - negative), axis=1)
                basic_loss = pos_dist - neg_dist + self.margin
                loss = K.maximum(basic_loss, 0.0)
                neg_dis = K.sum(anchor * negative, axis=1)
                dim = K.int_shape(y_pred)[1]
                gor = K.pow(K.mean(neg_dis), 2) + \
                    K.maximum(K.mean(K.pow(neg_dis, 2)) - 1.0 / dim, 0.0)
                return loss + (gor * self.alpha)

            loss = orthogonal_tloss(y_true, y_pred)
            o_self = only_self_regularization(y_pred)
            n_self = not_self_regularization(y_pred)
            b_self = both_self_regularization(y_pred)
            return loss + ((o_self + n_self + b_self) / 3)  # n_self

        def mse_loss(y_true, y_pred):
            def mse_loss(y_pred):
                anchor, positive, negative, anchor_sign3, _ = split_output(
                    y_pred)
                return keras.losses.mean_squared_error(anchor_sign3, anchor)
            loss = orthogonal_tloss(y_true, y_pred)
            mse_loss = mse_loss(y_pred)
            return loss + mse_loss

        lfuncs_dict = {'tloss': tloss,
                       'bayesian_tloss': bayesian_tloss,
                       'orthogonal_tloss': orthogonal_tloss,
                       'only_self_loss': only_self_loss,
                       'penta_loss': penta_loss}

        # compile and print summary
        self.__log.info('Loss function: %s' %
                        lfuncs_dict[self.loss_func].__name__)

        if self.learning_rate == 'auto':
            optimizer = keras.optimizers.Adam(learning_rate=MIN_LR)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=lfuncs_dict[self.loss_func],
            metrics=metrics)
        model.summary()

        # if pre-trained model is specified, load its weights
        self.model = model
        if load:
            self.model.load_weights(self.model_file)
        # this will be the encoder/transformer
        self.transformer = self.model.layers[-2]

    def find_lr(self, params, num_lr=5, generator=None):
        import matplotlib.pyplot as plt
        from scipy.stats import rankdata
        # Initialize model
        input_shape = (self.tr_shapes[0][1],)
        self.build_model(input_shape)

        # Find lr by grid search
        self.__log.info('Finding best lr')
        lr_iters = []
        lr_params = params.copy()
        lr_params['epochs'] = 1
        lrs = [1e-6, 1e-5, 1e-4]
        for lr in lrs:
            self.__log.info('Trying lr %s' % lr)
            lr_params['learning_rate'] = lr
            siamese = SiameseTriplets(
                self.model_dir, evaluate=True, plot=True, save_params=False,
                generator=generator, **lr_params)
            siamese.fit(save=False)
            h_file = os.path.join(
                self.model_dir, 'siamesetriplets_history.pkl')
            h_metrics = pickle.load(open(h_file, "rb"))
            loss = h_metrics['loss'][0]
            val_loss = h_metrics['val_loss'][0]
            acc = h_metrics['accTot'][0]
            val_acc = h_metrics['val_accTot'][0]
            lr_iters.append([loss, val_loss, val_acc])

        lr_iters = np.array(lr_iters)
        lr_scores = [rankdata(1 / col) if i > 1 else rankdata(col)
                     for i, col in enumerate(lr_iters.T)]
        lr_scores = np.mean(np.array(lr_scores).T, axis=1)
        lr_index = np.argmin(lr_scores)
        lr = lrs[lr_index]
        lr_results = {'lr_iters': lr_iters,
                      'lr_scores': lr_scores, 'lr': lr, 'lrs': lrs}

        fname = 'lr_score.pkl'
        pkl_file = os.path.join(self.model_dir, fname)
        pickle.dump(lr_results, open(pkl_file, "wb"),protocol=pickle.HIGHEST_PROTOCOL)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        ax = axes.flatten()
        log_lrs = np.log10(lrs)

        ax[0].set_title('Loss')
        ax[0].set_xlabel('lrs')
        ax[0].scatter(log_lrs, lr_iters[:, 0], label='train')
        ax[0].scatter(log_lrs, lr_iters[:, 1], label='test')
        ax[0].legend()

        ax[1].set_title('ValAccT')
        ax[1].set_xlabel('lrs')
        ax[1].scatter(log_lrs, lr_iters[:, 2], label='train')

        ax[2].set_title('Lr score')
        ax[2].set_xlabel('lrs')
        ax[2].scatter(log_lrs, lr_scores)
        fig.tight_layout()

        fname = 'lr_score.png'
        plot_file = os.path.join(self.model_dir, fname)
        plt.savefig(plot_file)
        plt.close()

        return lr

    def fit(self, monitor='val_loss', save=True):
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
            """
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan3 = np.isfinite(x3_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2, not_nan3)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            x3_data_transf = x3_data_transf[not_nan]
            """
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
            """
            not_nan1 = np.isfinite(x1_data_transf).any(axis=1)
            not_nan2 = np.isfinite(x2_data_transf).any(axis=1)
            not_nan3 = np.isfinite(x3_data_transf).any(axis=1)
            not_nan = np.logical_and(not_nan1, not_nan2, not_nan3)
            x1_data_transf = x1_data_transf[not_nan]
            x2_data_transf = x2_data_transf[not_nan]
            x3_data_transf = x3_data_transf[not_nan]
            """
            return x1_data_transf, x2_data_transf, x3_data_transf

        vsets = ['train_test', 'test_test']
        if self.evaluate and self.plot:
            # additional validation sets
            if "dataset_idx" in self.augment_kwargs:
                space_idx = self.augment_kwargs['dataset_idx']
                mask_fns = {
                    'ALL': None,
                    'NOT-SELF': partial(mask_exclude, space_idx),
                    'ONLY-SELF': partial(mask_keep, space_idx),
                }
            else:
                mask_fns = {
                    'ALL': None
                }
            validation_sets = list()

            for split in vsets:
                for set_name, mask_fn in mask_fns.items():
                    name = '_'.join([split, set_name])
                    shapes, dtypes, gen = TripletIterator.generator_fn(
                        self.traintest_file, split,
                        batch_size=self.batch_size,
                        shuffle=False,
                        replace_nan=self.replace_nan,
                        train=False,
                        augment_kwargs=self.augment_kwargs,
                        augment_fn=self.augment_fn,
                        mask_fn=mask_fn,
                        trim_mask=self.trim_mask,
                        sharedx=self.sharedx,
                        sharedx_trim=self.sharedx_trim,
                        onlyself_notself=self.onlyself_notself)
                    validation_sets.append((gen, shapes, name))
            additional_vals = AdditionalValidationSets(
                validation_sets, self.model, batch_size=self.batch_size,
                validation_steps=self.validation_steps)
            callbacks.append(additional_vals)

        class CustomEarlyStopping(EarlyStopping):

            def __init__(self,
                         monitor='val_loss',
                         min_delta=0,
                         patience=0,
                         verbose=0,
                         mode='auto',
                         baseline=None,
                         threshold=0,
                         restore_best_weights=False):
                super(EarlyStopping, self).__init__()

                self.monitor = monitor
                self.baseline = baseline
                self.patience = patience
                self.verbose = verbose
                self.min_delta = min_delta
                self.wait = 0
                self.stopped_epoch = 0
                self.restore_best_weights = restore_best_weights
                self.best_weights = None
                self.threshold = threshold

                if mode not in ['auto', 'min', 'max']:
                    mode = 'auto'

                if mode == 'min':
                    self.monitor_op = np.less
                elif mode == 'max':
                    self.monitor_op = np.greater
                else:
                    if 'acc' in self.monitor:
                        self.monitor_op = np.greater
                    else:
                        self.monitor_op = np.less

                if self.monitor_op == np.greater:
                    self.min_delta *= 1
                else:
                    self.min_delta *= -1

            def on_epoch_end(self, epoch, logs=None):
                current = self.get_monitor_value(logs)
                threshold = logs.get(self.monitor.replace('val_', ''))
                if current is None:
                    return

                if self.threshold > threshold:
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                elif self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    self.wait = 0
                    if self.restore_best_weights:
                        self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        if self.restore_best_weights:
                            if self.verbose > 0:
                                print('Restoring model weights from the end of '
                                      'the best epoch')
                            self.model.set_weights(self.best_weights)

        early_stopping = EarlyStopping(
            monitor=monitor,
            verbose=1,
            patience=self.patience,
            mode='min',
            restore_best_weights=True)
        if monitor or not self.evaluate:
            callbacks.append(early_stopping)

        # call fit and save model
        t0 = time()
        self.history = self.model.fit(
            self.tr_gen,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=self.val_gen,
            validation_steps=self.validation_steps,
            shuffle=True)
        self.time = time() - t0
        if save:
            self.model.save(self.model_file)
        if self.evaluate and self.plot:
            self.history.history.update(additional_vals.history)

        # check early stopping
        if early_stopping.stopped_epoch != 0:
            self.last_epoch = early_stopping.stopped_epoch - self.patience
        else:
            self.last_epoch = self.epochs

        # save and plot history
        history_file = os.path.join(
            self.model_dir, "%s_history.pkl" % self.name)
        pickle.dump(self.history.history, open(history_file, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
        history_file = os.path.join(self.model_dir, "history.png")
        anchor_file = os.path.join(self.model_dir, "anchor_distr.png")
        if self.evaluate and self.plot:
            self._plot_history(self.history.history, vsets, history_file)
        if self.onlyself_notself and self.plot:
            self._plot_anchor_dist(anchor_file)

    def predict(self, x_matrix, dropout_fn=None, dropout_samples=10, cp=False):
        """Do predictions.

        prediction_file(str): Path to input file containing Xs.
        split(str): which split to predict.
        batch_size(int): batch size for prediction.
        """

        # apply input scaling
        if hasattr(self, 'scaler'):
            # scaler has already been trimmed
            scaled = self.scaler.transform(x_matrix)
        else:
            scaled = x_matrix

        # apply trimming of input matrix
        if self.trim_mask is not None:
            trimmed = scaled[:, np.repeat(self.trim_mask, 128)]
        else:
            trimmed = scaled

        # load model if not alredy there
        if self.model is None:
            self.build_model((trimmed.shape[1],), load=True, cp=cp)

        # get rid of NaNs
        no_nans = np.nan_to_num(trimmed)
        # get default dropout function
        if dropout_fn is None:
            return self.transformer.predict(no_nans)
        # sample with dropout (repeat input)
        samples = list()
        for i in range(dropout_samples):
            dropped_ds = dropout_fn(no_nans)
            no_nans_drop = np.nan_to_num(dropped_ds)
            samples.append(self.transformer.predict(no_nans_drop))
        samples = np.vstack(samples)
        samples = samples.reshape(
            no_nans.shape[0], dropout_samples, samples.shape[1])
        return samples

    def _plot_history(self, history, vsets, destination):
        """Plot history.

        history(dict): history result from Keras fit method.
        destination(str): path to output file.
        """
        import matplotlib.pyplot as plt

        metrics = sorted(list({k.split('_')[-1] for k in history}))

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

        def sim(a, b):
            return -(cosine(a, b) - 1)

        # Need to create a new train_train generator without train=False
        tr_shape_type_gen = TripletIterator.generator_fn(
            self.traintest_file,
            'train_train',
            batch_size=self.batch_size,
            shuffle=False,
            replace_nan=self.replace_nan,
            train=False,
            augment_fn=self.augment_fn,
            augment_kwargs=self.augment_kwargs,
            sharedx=self.sharedx,
            onlyself_notself=self.onlyself_notself)

        tr_gen = tr_shape_type_gen[2]()

        if self.evaluate:
            trval_shape_type_gen = TripletIterator.generator_fn(
                self.traintest_file,
                'train_test',
                batch_size=self.batch_size,
                shuffle=False,
                replace_nan=self.replace_nan,
                train=False,
                augment_fn=self.augment_fn,
                augment_kwargs=self.augment_kwargs,
                sharedx=self.sharedx,
                onlyself_notself=self.onlyself_notself)
            trval_gen = trval_shape_type_gen[2]()

            val_shape_type_gen = TripletIterator.generator_fn(
                self.traintest_file,
                'test_test',
                batch_size=self.batch_size,
                shuffle=False,
                replace_nan=self.replace_nan,
                train=False,
                augment_fn=self.augment_fn,
                augment_kwargs=self.augment_kwargs,
                sharedx=self.sharedx,
                onlyself_notself=self.onlyself_notself)
            val_gen = val_shape_type_gen[2]()

            vset_dict = {'train_train': tr_gen,
                         'train_test': trval_gen, 'test_test': val_gen}
        else:
            vset_dict = {'train_train': tr_gen}

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
                if len(anchors) >= 10000:
                    break
            anchors = np.array(anchors)
            positives = np.array(positives)
            negatives = np.array(negatives)
            labels = np.array(labels)

            ap_dists = np.linalg.norm(anchors - positives, axis=1)
            an_dists = np.linalg.norm(anchors - negatives, axis=1)

            mask_e = labels == 0
            mask_m = labels == 1
            mask_h = labels == 2

            ax.set_title('Euclidean ' + vset)
            sns.kdeplot(ap_dists[mask_e], label='pos_e',
                        ax=ax, color='limegreen')
            sns.kdeplot(ap_dists[mask_m], label='pos_m',
                        ax=ax, color='forestgreen')
            sns.kdeplot(ap_dists[mask_h], label='pos_h',
                        ax=ax, color='darkgreen')

            sns.kdeplot(an_dists[mask_e], label='neg_e', ax=ax, color='salmon')
            sns.kdeplot(an_dists[mask_m], label='neg_m', ax=ax, color='red')
            sns.kdeplot(an_dists[mask_h], label='neg_h',
                        ax=ax, color='darkred')

            ax.legend()

            ax = axes[i]
            i += 1

            ax.scatter(ap_dists[mask_e][:1000], an_dists[mask_e][:1000],
                       label='easy', color='green', s=2)
            ax.scatter(ap_dists[mask_m][:1000], an_dists[mask_m][:1000],
                       label='medium', color='goldenrod', s=2, alpha=0.7)
            ax.scatter(ap_dists[mask_h][:1000], an_dists[mask_h][:1000],
                       label='hard', color='red', s=2, alpha=0.7)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            ax.set_xlabel('Euc dis positives')
            ax.set_ylabel('Euc dis negatives')

            ax = axes[i]
            i += 1

            ap_sim = np.array([sim(anchors[i], positives[i])
                               for i in range(len(anchors))])
            an_sim = np.array([sim(anchors[i], negatives[i])
                               for i in range(len(anchors))])

            ax.set_title('Cosine ' + vset)
            sns.kdeplot(ap_sim[mask_e], label='pos_e',
                        ax=ax, color='limegreen')
            sns.kdeplot(ap_sim[mask_m], label='pos_m',
                        ax=ax, color='forestgreen')
            sns.kdeplot(ap_sim[mask_h], label='pos_h',
                        ax=ax, color='darkgreen')
            plt.xlim(-1, 1)

            sns.kdeplot(an_sim[mask_e], label='neg_e', ax=ax, color='salmon')
            sns.kdeplot(an_sim[mask_m], label='neg_m', ax=ax, color='red')
            sns.kdeplot(an_sim[mask_h], label='neg_h', ax=ax, color='darkred')
            plt.xlim(-1, 1)
            ax.legend()

            ax = axes[i]
            i += 1

            ax.scatter(ap_sim[mask_e][:1000], an_sim[mask_e][:1000],
                       label='easy', color='green', s=2)
            ax.scatter(ap_sim[mask_m][:1000], an_sim[mask_m][:1000],
                       label='medium', color='goldenrod', s=2, alpha=0.7)
            ax.scatter(ap_sim[mask_h][:1000], an_sim[mask_h][:1000],
                       label='hard', color='red', s=2, alpha=0.7)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            ax.set_xlabel('Cos sim positives')
            ax.set_ylabel('Cos sim negatives')

        plt.savefig(plot_file)
        plt.close()


class AdditionalValidationSets(Callback):

    def __init__(self, validation_sets, model, verbose=1, batch_size=None,
                 validation_steps=None):
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
        self.validation_steps = validation_steps
        if self.validation_steps is None:
            self.validation_steps = np.ceil(val_shapes[0][0] / self.batch_size)
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.set_model( model )

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
            results = self.model.evaluate(
                val_gen(),
                steps=self.validation_steps,
                verbose=self.verbose)

            for i, result in enumerate(results):
                name = '_'.join([val_set_name, self.model.metrics_names[i]])
                self.history.setdefault(name, []).append(result)
