import functools
import adanet
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import layers
#from tensorflow.compat.v1.layers import Dense, Dropout
from chemicalchecker.util import logged


class NanMaskingLayer(layers.Layer):

    def __init__(self, mask_value=0.0):
        super(NanMaskingLayer, self).__init__()
        self.mask_value = mask_value

    def call(self, input):
        nan_idxs = tf.math.is_nan(input)
        replace = tf.ones_like(input) * self.mask_value
        return tf.where(nan_idxs, replace, input)


@logged
class ExtendDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, optimizer, layer_sizes, num_layers, layer_block_size,
                 learn_mixture_weights, dropout, seed, activation,
                 previous_ensemble, input_shape, nan_mask_value=0.0):
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
          An instance of `ExtendDNNBuilder`.
        """

        self._optimizer = optimizer
        self._layer_sizes = layer_sizes
        self._num_layers = num_layers
        self._layer_block_size = layer_block_size
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed
        self._dropout = dropout
        self._activation = activation
        self._input_shape = input_shape
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
        for layer_size in self._layer_sizes:
            last_layer = layers.Dense(
                layer_size * self._layer_block_size,
                activation=self._activation)(last_layer)
            last_layer = layers.Dropout(
                rate=self._dropout,
                seed=self._seed)(last_layer, training=training)(last_layer)
        logits = layers.Dense(logits_dimension)(last_layer)

        shared_tensors = {
            "num_layers": tf.constant(self._num_layers),
            "layer_sizes": tf.constant(self._layer_sizes),
        }
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=self._measure_complexity(),
            shared=shared_tensors)

    def _measure_complexity(self):
        """Approximates Rademacher complexity as square-root of the depth."""
        return tf.sqrt(tf.cast(tf.math.reduce_sum(self._layer_sizes), dtype=tf.float32))

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
        if len(self._layer_sizes) == 0:
            layer_size_str = '0'
        else:
            layer_size_str = '_'.join([str(x) for x in self._layer_sizes])
        return "dnn_{}_layer_{}_nodes".format(self._num_layers, layer_size_str)


@logged
class ExtendDNNGenerator(adanet.subnetwork.Generator):
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
                 initial_architecture=[1],
                 extension_step=1):
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
        self._layer_block_size = layer_size
        self._layer_sizes = initial_architecture
        self._extension_step = extension_step
        self._num_layers = len(initial_architecture)
        self._dnn_builder_fn = functools.partial(
            ExtendDNNBuilder,
            optimizer=optimizer,
            dropout=dropout,
            input_shape=input_shape,
            nan_mask_value=nan_mask_value,
            activation=activation,
            layer_block_size=layer_size,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number,
                            previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        seed = self._seed
        if seed is not None:
            seed += iteration_number
        num_layers = self._num_layers
        layer_sizes = self._layer_sizes
        # take the maximum depth reached in previous iterations + 1
        if previous_ensemble:
            last_subnetwork = previous_ensemble.weighted_subnetworks[
                -1].subnetwork
            shared_tensors = last_subnetwork.shared
            num_layers = tf.get_static_value(
                shared_tensors["num_layers"])
            layer_sizes = list(tf.get_static_value(
                shared_tensors["layer_sizes"]))
        # at each iteration try exdending any of the existing layers (width)
        candidates = list()
        if iteration_number != 0:
            for extend_layer in range(num_layers):
                new_sizes = layer_sizes[:]
                new_sizes[extend_layer] += self._extension_step
                candidates.append(
                    self._dnn_builder_fn(
                        num_layers=num_layers,
                        layer_sizes=new_sizes,
                        seed=seed,
                        previous_ensemble=previous_ensemble))
            # try adding a new layer (depth)
            candidates.append(
                self._dnn_builder_fn(
                    num_layers=num_layers + self._extension_step,
                    layer_sizes=layer_sizes + [1] * self._extension_step,
                    seed=seed,
                    previous_ensemble=previous_ensemble))
        # keep the un-extended candidate
        candidates.append(
            self._dnn_builder_fn(
                num_layers=num_layers,
                layer_sizes=layer_sizes,
                seed=seed,
                previous_ensemble=previous_ensemble))
        return candidates
