import functools
import adanet
import tensorflow.compat.v1 as tf


class SimpleCNNBuilder(adanet.subnetwork.Builder):
    """Builds a CNN subnetwork for AdaNet."""

    def __init__(self, learning_rate, max_iteration_steps, seed):
        """Initializes a `SimpleCNNBuilder`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `SimpleCNNBuilder`.
        """
        self._learning_rate = learning_rate
        self._max_iteration_steps = max_iteration_steps
        self._seed = seed

    def build_subnetwork(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        images = tf.to_float(features['x'])
        print "***********", images.shape
        kernel_initializer = tf.initializers.he_normal(seed=self._seed)
        x = tf.layers.Conv2D(
            input_shape=images.shape,
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=kernel_initializer)(images)
        x = tf.layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = tf.layers.Flatten()(x)
        x = tf.layers.Dense(
            units=64,
            activation="relu",
            kernel_initializer=kernel_initializer)(x)

        # The `Head` passed to adanet.Estimator will apply the softmax
        # activation.
        logits = tf.layers.Dense(
            units=logits_dimension,
            activation=None,
            kernel_initializer=kernel_initializer)(x)

        # Use a constant complexity measure, since all subnetworks have same
        # architecture and hyperparameters.
        complexity = tf.constant(1)

        return adanet.Subnetwork(
            last_layer=x,
            logits=logits,
            complexity=complexity,
            persisted_tensors={})

    def build_subnetwork_train_op(self,
                                  subnetwork,
                                  loss,
                                  var_list,
                                  labels,
                                  iteration_step,
                                  summary,
                                  previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""

        # Momentum optimizer with cosine learning rate decay works well with
        # CNNs.
        learning_rate = tf.train.cosine_decay(
            learning_rate=self._learning_rate,
            global_step=iteration_step,
            decay_steps=self._max_iteration_steps)
        optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
        # NOTE: The `adanet.Estimator` increments the global step.
        return optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                       iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""
        return tf.no_op("mixture_weights_train_op")

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        return "simple_cnn"


class SimpleCNNGenerator(adanet.subnetwork.Generator):
    """Generates a `SimpleCNN` at each iteration.
    """

    def __init__(self, learning_rate, max_iteration_steps, seed=None):
        """Initializes a `Generator` that builds `SimpleCNNs`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `Generator`.
        """
        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            SimpleCNNBuilder,
            learning_rate=learning_rate,
            max_iteration_steps=max_iteration_steps)

    def generate_candidates(self, previous_ensemble, iteration_number,
                            previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        seed = self._seed
        # Change the seed according to the iteration so that each subnetwork
        # learns something different.
        if seed is not None:
            seed += iteration_number
        return [self._dnn_builder_fn(seed=seed)]
