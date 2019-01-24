import functools
import adanet
import tensorflow as tf
from chemicalchecker.util import logged


@logged
class StackDNNBuilder(adanet.subnetwork.Builder):
    """Builds a DNN subnetwork for AdaNet."""

    def __init__(self, optimizer, layer_size, num_layers,
                 learn_mixture_weights, seed, activation, previous_ensemble):
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
          seed: A random seed.

        Returns:
          An instance of `StackDNNBuilder`.
        """

        self._optimizer = optimizer
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._learn_mixture_weights = learn_mixture_weights
        self._seed = seed
        self._activation = activation
        self._previous_ensemble = previous_ensemble

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble):
        """See `adanet.subnetwork.Builder`."""

        input_layer = tf.to_float(features['x'])
        kernel_initializer = tf.glorot_uniform_initializer(seed=self._seed)
        last_layer = input_layer
        persisted_tensors = dict()
        for i in range(self._num_layers):
            last_layer = tf.layers.dense(
                last_layer,
                units=self._layer_size,
                activation=self._activation,
                kernel_initializer=kernel_initializer)
            hidden_layer_key = "hidden_layer_{}".format(i)
            self.__log.debug("hidden_layer_key: %s" % hidden_layer_key)
            if previous_ensemble:
                last_subnetwork = self._previous_ensemble.weighted_subnetworks[
                    -1].subnetwork
                persisted_tensors = last_subnetwork.persisted_tensors
                self.__log.debug("persisted_tensors: %s" %
                                 str(persisted_tensors.keys()))
                if hidden_layer_key in persisted_tensors:
                    last_layer = tf.concat([persisted_tensors[
                        hidden_layer_key], last_layer], axis=1)
            # Store hidden layer outputs for subsequent iterations.
            persisted_tensors[hidden_layer_key] = last_layer
        # update the num layers in persistent tensor
        persisted_tensors.update({"num_layers": tf.constant(self._num_layers)})

        self.__log.debug("persisted_tensors DONE: %s" %
                         str(persisted_tensors.keys()))
        logits = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=kernel_initializer)
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


@logged
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
                 layer_size=32,
                 learn_mixture_weights=False,
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
          seed: A random seed.

        Returns:
          An instance of `Generator`.
        """

        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            StackDNNBuilder,
            optimizer=optimizer,
            layer_size=layer_size,
            activation=activation,
            learn_mixture_weights=learn_mixture_weights)

    def generate_candidates(self, previous_ensemble, iteration_number,
                            previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        seed = self._seed
        if seed is not None:
            seed += iteration_number
        # initial number of layer (0 == linear)
        num_layers = 0
        # take the maximum depth reached in previous iterations + 1
        if previous_ensemble:
            last_subnetwork = previous_ensemble.weighted_subnetworks[
                -1].subnetwork
            persisted_tensors = last_subnetwork.persisted_tensors
            num_layers = tf.contrib.util.constant_value(
                persisted_tensors["num_layers"])
        return [
            self._dnn_builder_fn(
                num_layers=num_layers, seed=seed,
                previous_ensemble=previous_ensemble),
            self._dnn_builder_fn(
                num_layers=num_layers + 1,
                seed=seed,
                previous_ensemble=previous_ensemble),
        ]
