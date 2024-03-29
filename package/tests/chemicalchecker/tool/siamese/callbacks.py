from tensorflow import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pickle
from scipy.stats import linregress


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
            A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
            A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
            A cycle that scales initial amplitude by gamma**(cycle iterations) at each
            cycle iteration.
    For more detail, please see paper.

    # Example
            ```python
                    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                                            step_size=2000., mode='triangular')
                    model.fit(X_train, Y_train, callbacks=[clr])
            ```

    Class also supports custom scaling functions:
            ```python
                    clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
                    clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                                            step_size=2000., scale_fn=clr_fn,
                                                            scale_mode='cycle')
                    model.fit(X_train, Y_train, callbacks=[clr])
            ```
    # Arguments
            base_lr: initial learning rate which is the
                    lower boundary in the cycle.
            max_lr: upper boundary in the cycle. Functionally,
                    it defines the cycle amplitude (max_lr - base_lr).
                    The lr at any cycle is the sum of base_lr
                    and some scaling of the amplitude; therefore
                    max_lr may not actually be reached depending on
                    scaling function.
            step_size: number of training iterations per
                    half cycle. Authors suggest setting step_size
                    2-8 x training iterations in epoch.
            mode: one of {triangular, triangular2, exp_range}.
                    Default 'triangular'.
                    Values correspond to policies detailed above.
                    If scale_fn is not None, this argument is ignored.
            gamma: constant in 'exp_range' scaling function:
                    gamma**(cycle iterations)
            scale_fn: Custom scaling policy defined by a single
                    argument lambda function, where
                    0 <= scale_fn(x) <= 1 for all x >= 0.
                    mode paramater is ignored
            scale_mode: {'cycle', 'iterations'}.
                    Defines whether scale_fn is evaluated on
                    cycle number or cycle iterations (training
                    iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class LearningRateFinder:

    def __init__(self, model, stopFactor=4, beta=0.98):
        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta

        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []

        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
                       "Iterator", "Sequence"]

        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss

        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return

        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR, epochs=None,
             stepsPerEpoch=None, batchSize=32, sampleSize=2048,
             verbose=1):
        # reset our class-specific variables
        self.reset()

        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch

        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
                                  self.on_batch_end(batch, logs))

        self.model.fit_generator(
            trainData,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=[callback],
            shuffle=True)

        # restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)

    def find_bounds(self, min_x=-7, max_x=1, interval=0.5):
        x = np.log10(self.lrs)
        y = self.losses
        idx = np.argmin(y)
        ub = x[idx]
        plat = int(idx*interval)+1
        y_plat = np.mean(y[:plat])
        slope = int(idx*(1-interval))+1
        reg = linregress(x[slope:idx], y[slope:idx])
        lb = (y_plat - reg.intercept)/reg.slope
        return np.max([lb,min_x]), np.min([ub,max_x])

    def plot_loss(self, min_lr, max_lr, plot_file, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = np.log10(self.lrs[skipBegin:-skipEnd])
        losses = self.losses[skipBegin:-skipEnd]

        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.axvline(min_lr, color='black')
        plt.axvline(max_lr, color='black')
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)
        plt.savefig(plot_file)

    def save_loss_evolution(self, fname):
        evolution = {'lrs': self.lrs, 'losses': self.losses}
        pickle.dump(evolution, open(fname, "wb"))
