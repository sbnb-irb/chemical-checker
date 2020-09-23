import functools
import adanet
import tensorflow as tf

from tensorflow import layers
from tensorflow.layers import Dense, Dropout

from chemicalchecker.util import logged


class NanMaskingLayer(layers.Layer):

    def __init__(self, mask_value=0.0):
        super(NanMaskingLayer, self).__init__()
        self.mask_value = mask_value

    def call(self, input):
        nan_idxs = tf.is_nan(input)
        replace = tf.ones_like(input) * self.mask_value
        return tf.where(nan_idxs, replace, input)


@logged
class StackDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, optimizer, layer_size, num_layers,
                 learn_mixture_weights, dropout, seed, activation,
                 input_shape, nan_mask_value=0.0):
        """Initializes a `_DNNBuilder`.

        Args:
          optimizer: An `Optimizer` instance for training both the subnetwork
            and the mixture weights.
          layer_size: The number of nodes to output at each hidden layer.
          num_layers: The number of hidden layers.
          learn_mixture_weights: Whether to solve a learning problem to find
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a
            no_op for the mixture weight train op.
          dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would
            drop out 10% of input units.
          activation: The activation function to be used.
          seed: A random seed.

        Returns:
          An instance of `StackDNNBuilder`.
        """

        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed
        self._dropout = dropout
        self._input_shape = input_shape
        self._activation = activation
        self._nan_mask_value = nan_mask_value

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""

        input_layer = tf.cast(features['x'], tf.float32)
        # forcing to input shape as dataset uses tf.py_func (loosing shape)
        input_layer = tf.reshape(features['x'], [-1, self._input_shape])
        last_layer = input_layer
        if self._nan_mask_value is not None:
            last_layer = NanMaskingLayer(self._nan_mask_value)(last_layer)
        for _ in range(self._num_layers):
            last_layer = Dense(
                self._layer_size,
                activation=self._activation)(last_layer)
            last_layer = Dropout(
                rate=self._dropout,
                seed=self._seed)(last_layer, training=training)

        logits = Dense(units=logits_dimension)(last_layer)

        persisted_tensors = {"num_layers": tf.constant(self._num_layers)}
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=self._measure_complexity(),
            persisted_tensors=persisted_tensors)

    def _measure_complexity(self):
        """Approximates Rademacher complexity as square-root of the depth."""
        return tf.sqrt(tf.to_float(self._num_layers))

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                  iteration_step, summary, previous_ensemble):
        """See `adanet.subnetwork.Builder`."""
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                       iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""

        if not self._learn_mixture_weights:
            return tf.no_op()
        return self._optimizer.minimize(loss=loss, var_list=var_list)

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""

        if self._num_layers == 0:
            # A DNN with no hidden layers is a linear model.
            return "linear"
        return "{}_layer_dnn".format(self._num_layers)


class StackDNNGenerator(adanet.subnetwork.Generator):
    """Generates a two DNN subnetworks at each iteration.

    The first DNN has an identical shape to the most recently added subnetwork
    in `previous_ensemble`. The second has the same shape plus one more dense
    layer on top. This is similar to the adaptive network presented in Fig2
    [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
    connections to hidden layers of networks from previous iterations.
    """

    def __init__(self,
                 optimizer,
                 input_shape,
                 nan_mask_value=0.0,
                 layer_size=32,
                 learn_mixture_weights=False,
                 dropout=0.0,
                 activation=tf.nn.relu,
                 seed=None,
                 **kwargs):
        """Initializes a DNN `Generator`.

        Args:
          optimizer: An `Optimizer` instance for training both the subnetwork
            and the mixture weights.
          layer_size: Number of nodes in each hidden layer of the subnetwork
            candidates. Note that this parameter is ignored in a DNN with no
            hidden layers.
          learn_mixture_weights: Whether to solve a learning problem to find
            best mixture weights, or use their default value according to the
            mixture weight type. When `False`, the subnetworks will return a
            no_op for the mixture weight train op.
          dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would
            drop out 10% of input units.
          activation: The activation function to be used.
          seed: A random seed.

        Returns:
          An instance of `Generator`.
        """

        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            StackDNNBuilder,
            optimizer=optimizer,
            layer_size=layer_size,
            dropout=dropout,
            input_shape=input_shape,
            nan_mask_value=nan_mask_value,
            activation=activation,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number,
                            previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""

        num_layers = 0
        seed = self._seed
        if previous_ensemble:
            num_layers = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[
                    -1].subnetwork.persisted_tensors["num_layers"])
        if seed is not None:
            seed += iteration_number
        return [
            self._dnn_builder_fn(num_layers=num_layers, seed=seed),
            self._dnn_builder_fn(num_layers=num_layers + 1, seed=seed),
        ]
