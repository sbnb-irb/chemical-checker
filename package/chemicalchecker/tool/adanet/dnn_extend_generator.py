import functools
import adanet
import tensorflow as tf
import numpy as np
from chemicalchecker.util import logged


@logged
class ExtendDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, optimizer, layer_sizes, num_layers, layer_block_size,
                 learn_mixture_weights, dropout, seed, activation, previous_ensemble):
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

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        input_layer = tf.to_float(features['x'])
        kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        last_layer = input_layer
        for layer_size in self._layer_sizes:
            last_layer = tf.layers.dense(
                last_layer,
                units=layer_size * self._layer_block_size,
                activation=self._activation,
                kernel_initializer=kernel_initializer)
            last_layer = tf.layers.dropout(
                last_layer,
                rate=self._dropout,
                seed=self._seed,
                training=training)
        logits = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=kernel_initializer)

        persisted_tensors = {
            "num_layers": tf.constant(self._num_layers),
            "layer_sizes": tf.constant(self._layer_sizes),
        }
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=self._measure_complexity(),
            persisted_tensors=persisted_tensors)

    def _measure_complexity(self):
        """Approximates Rademacher complexity as square-root of the depth."""
        #depth_cmpl = np.sqrt(float(self._num_layers))
        #max_width_cmpl = np.sqrt(float(max(self._layer_sizes)))
        total_blocks_cmpl = np.sqrt(float(sum(self._layer_sizes)))
        #self.__log.debug("\n\n***** COMPLEXITY\ndepth_cmpl: %s\max_width_cmpl %s\total_blocks_cmpl %s\n\n", depth_cmpl, max_width_cmpl, total_blocks_cmpl)
        return total_blocks_cmpl

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
                 layer_size=32,
                 learn_mixture_weights=False,
                 dropout=0.0,
                 activation=tf.nn.relu,
                 seed=None):
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
        self._dnn_builder_fn = functools.partial(
            ExtendDNNBuilder,
            optimizer=optimizer,
            dropout=dropout,
            activation=activation,
            layer_block_size=layer_size,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number,
                            previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        seed = self._seed
        if seed is not None:
            seed += iteration_number
        # start with single layer
        num_layers = 0
        layer_sizes = []
        # take the maximum depth reached in previous iterations + 1
        if previous_ensemble:
            last_subnetwork = previous_ensemble.weighted_subnetworks[
                -1].subnetwork
            persisted_tensors = last_subnetwork.persisted_tensors
            num_layers = tf.contrib.util.constant_value(
                persisted_tensors["num_layers"])
            layer_sizes = list(tf.contrib.util.constant_value(
                persisted_tensors["layer_sizes"]))
        # at each iteration we want to check if exdending any of the
        # existing layes is good
        candidates = list()
        for extend_layer in range(num_layers):
            new_sizes = layer_sizes[:]
            new_sizes[extend_layer] += 1
            candidates.append(
                self._dnn_builder_fn(
                    num_layers=num_layers,
                    layer_sizes=new_sizes,
                    seed=seed,
                    previous_ensemble=previous_ensemble))
        # also check if it's worth adding a new layer
        candidates.append(
            self._dnn_builder_fn(
                num_layers=num_layers + 1,
                layer_sizes=layer_sizes + [1],
                seed=seed,
                previous_ensemble=previous_ensemble))
        # also keep the un-extended candidate
        candidates.append(
            self._dnn_builder_fn(
                num_layers=num_layers,
                layer_sizes=layer_sizes,
                seed=seed,
                previous_ensemble=previous_ensemble))
        return candidates
