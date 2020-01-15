from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chemicalchecker.util import logged


@logged
class Siamese(object):


    def __init__(self, models_path, **kwargs):
        # Models path
        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            self.__log.warning(
                "Specified models directory does not exist: %s",
                self.models_path)
            os.mkdir(self.models_path)

        self.epochs = int(kwargs.get("epochs", 200))
        self.batch_size = int(kwargs.get("batch_size", 128))

        self.siaclassifier_model_file = os.path.join(self.models_path, "siaclassifier.h5")
        self.siamese = None



    def fit(self, data_path):
        """Take data .h5 and learn an siamese classifier.

        Args:
            data_path(string): a path to .h5 file.
        """

        self.data_path = data_path

        config = tf.ConfigProto(device_count={'GPU': 1})
        session = tf.Session(config=config)
        K.set_session(session)

        if not os.path.isfile(self.data_path) or self.data_path[-3:] != '.h5':
            raise Exception("Input data needs to be a H5 file")

        self.traintest_file = os.path.join(self.models_path, 'traintest.h5')

        self.input_dimension = 


        self.training_generator = DataGenerator(partition['train'], labels, **params)
        self.test_generator = DataGenerator(partition['validation'], labels, **params)

        self.input_shape = #INPUTSHAPE
        
        base_network = create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        self.siamese = Model([input_a, input_b], distance)

        self.__log.debug("Num of layers: %d" % len(layer_sizes))

        rms = RMSprop()
        self.siamese.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

        history = self.siamese.fit_generator(generator=training_generator,
                            batch_size=128,
                            epochs=epochs,
                            validation_data=validation_generator,
                            epochs=epochs)

        self.siamese.save(self.autoencoder_model_file)

        self._plot_history(history, os.path.join(self.models_path, "siamese_validation_plot.png"))


    @staticmethod
    def create_base_network(input_shape):
        '''Create network architecture'''

        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        m = Model(input, x)
        m.name = 'base_network' 
        return m

    def _plot_history(self, history, destination=None):
        plt.figure(figsize=(4, 4), dpi=600)
        plt.plot(history.history["loss"], label="Train loss", lw=1)
        plt.plot(history.history["val_loss"], label="Test loss", lw=1)
        plt.legend(loc='best')
        if destination is not None:
            plt.savefig(destination)


    def predict(self, data_path, dest_file, chunk_size=1000, input_dataset='V'):
        """Take data .h5 and produce an encoded data.

        Args:
            data_path(string): a path to input .h5 file.
            dest_file(string): a path to output .h5 file.
            chunk_size(int): numbe rof inputs at each prediction.
            dataset(string): The name of the dataset in the .h5 file to encode(default: 'V')
        """

        self.siamese = load_model(self.autoencoder_model_file)
        self.transformer = self.siamese.get_layer('base_network')
        
        data = # read_datapath
        with h5py.File(dest_file, "w") as results:
            #create datasets
            #predict by chunks
            #savefile



def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


    
