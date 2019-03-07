"""Utility for plotting from data.

Basically every Chemical Checker dataset can produce one or more plots.
This class performs offers a shared functions (e.g. color codes) to produce
different kind of plots.
"""
from __future__ import division

import os
import sys
import math
import h5py
import random
import functools
import numpy as np
import pandas as pd
from numpy import matlib
from scipy import stats
from scipy.stats import ks_2samp, fisher_exact, gaussian_kde
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.metrics.pairwise import cosine_distances

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from chemicalchecker.util import logged

random.seed(42)
np.random.seed(42)


def skip_on_exception(function):
    """Assist in skipping failing plots gracefully."""
    @logged
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            wrapper._log.error(err)
            return None
    return wrapper


@logged
class Plot():
    """Produce different kind of plots."""

    def __init__(self, dataset, plot_path, validation_path=None):
        """Initialize the Plot object.

        Produce all kind of plots and data associated

        Args:
            dataset(str): A Dataset object or the dataset code.
            plot_path(str): Final destination for the new plot and other stuff.
            validation_path(str): Directory where validation stats are found.
        """
        if not os.path.isdir(plot_path):
            raise Exception("Folder to save plots does not exist")
        if hasattr(dataset, 'code'):
            dataset_code = dataset.code
        else:
            dataset_code = dataset
        self.__log.debug('Plots for %s saved to %s',
                         dataset_code, plot_path)
        self.plot_path = plot_path
        self.validation_path = validation_path
        self.dataset_code = dataset_code
        self.color = self._coord_color(dataset_code)

    def _elbow(self, curve):
        nPoints = len(curve)
        allCoord = np.vstack((range(nPoints), curve)).T
        np.array([range(nPoints), curve])
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(
            vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = np.argmax(distToLine)
        return idxOfBestPoint

    def _rgb2hex(self, r, g, b):
        return '#%02x%02x%02x' % (r, g, b)

    def _coord_color(self, coordinate):
        A = self._rgb2hex(250, 100, 80)
        B = self._rgb2hex(200, 100, 225)
        C = self._rgb2hex(80, 120, 220)
        D = self._rgb2hex(120, 180, 60)
        E = self._rgb2hex(250, 150, 50)
        if coordinate[0] == 'A':
            return A
        if coordinate[0] == 'B':
            return B
        if coordinate[0] == 'C':
            return C
        if coordinate[0] == 'D':
            return D
        if coordinate[0] == 'E':
            return E

    def clustering_plot(self, Nc, A, B, C):

        sns.set_style("white")

        plt.figure(figsize=(4, 4), dpi=600)

        fig = plt.subplot(111)
        plt.plot(Nc, A, color=self.color, lw=1, linestyle=':')
        plt.plot(Nc, B, color=self.color, lw=1, linestyle='--')
        plt.plot(Nc, C, color=self.color, lw=2, linestyle='-')

        kidx = np.argmax(C)
        k = Nc[kidx]
        plt.plot([k, k], [0, 1], color=self.color, lw=1, linestyle='-')

        plt.ylim(0, 1)
        plt.xlim(np.min(Nc), np.max(Nc))
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Cluster measures")

        plt.title("k: %d clusters" % k)

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/kmeans_kselect.png" % self.plot_path)

        with open("%s/kmeans_kselect.tsv" % self.plot_path, "w") as f:
            for i in range(len(Nc)):
                f.write("%d\t%f\t%f\t%f\n" % (Nc[i], A[i], B[i], C[i]))

        return k

    def variance_plot(self, exp_var_ratios, variance_cutoff=0.9):

        # Set the variance-explained cutoff.

        sns.set_style("white")

        cumsum = np.cumsum(exp_var_ratios)

        plt.figure(figsize=(4, 4), dpi=600)

        fig = plt.subplot(111)
        x = [i for i in range(len(cumsum) + 1)]
        y = [0] + list(cumsum)
        plt.plot(x, y, color=self.color, lw=2)

        # Varcut
        varcut = variance_cutoff
        for i in range(len(cumsum)):
            if cumsum[i] > varcut:
                break
        i90 = i
        plt.plot([0., i90 + 1], [varcut, varcut],
                 color=self.color, linestyle="-")
        plt.plot([i90 + 1, i90 + 1], [0, varcut],
                 color=self.color, linestyle="-")
        plt.scatter([i90 + 1], [varcut], color="white",
                    edgecolor=self.color, lw=1.5, zorder=3, s=50)

        # Elbow point
        curve = cumsum[:i90 + 1]
        ielb = self._elbow(curve)
        elb = cumsum[ielb]
        plt.plot([0., ielb + 1], [elb, elb], color=self.color, linestyle="--")
        plt.plot([ielb + 1, ielb + 1], [0, elb],
                 color=self.color, linestyle="--")
        plt.scatter([ielb + 1], [elb], color="white",
                    edgecolor=self.color, lw=1.5, zorder=3, s=50)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, len(cumsum))
        plt.xlabel("Latent variables")
        plt.ylabel("Proportion of variance explained")

        plt.title("%.1f: %d, elbow: %d" % (varcut, i90 + 1, ielb + 1))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/variance_explained.png" % self.plot_path)

        with open("%s/variance_explained.tsv" % self.plot_path, "w") as f:
            for i in range(len(x)):
                f.write("%f\t%f\n" % (x[i], y[i]))

        return i90, ielb

    # Validate using moa and KS test

    def _for_the_validation(self, inchikey_dict, prefix, inchikey_mappings=None):

        f = open(self.validation_path + "/%s_validation.tsv" % prefix, "r")
        S = set()
        D = set()
        inchikeys = set()
        for l in f:
            l = l.rstrip("\n").split("\t")
            l0 = l[0]
            l1 = l[1]
            if inchikey_mappings is not None:
                if l0 in inchikey_mappings:
                    l0 = inchikey_mappings[l0]
                if l1 in inchikey_mappings:
                    l1 = inchikey_mappings[l1]
            inchikeys.update([l0, l1])
            if int(l[2]) == 1:
                S.update([(l0, l1)])
            else:
                if len(D) < 100000:
                    D.update([(l0, l1)])
                else:
                    pass
        f.close()

        d = {}
        for inchikey in inchikeys:
            try:
                d[inchikey] = inchikey_dict[inchikey]
            except:
                continue
        inchikeys = inchikeys.intersection(d.keys())
        S = set([x for x in S if x[0] in inchikeys and x[1] in inchikeys])
        D = set([x for x in D if x[0] in inchikeys and x[1] in inchikeys])
        d = dict((k, d[k]) for k in inchikeys)

        return S, D, d

    def label_validation(self, inchikey_lab, label_type, prefix="moa", inchikey_mappings=None):

        S, D, d = self._for_the_validation(
            inchikey_lab, prefix, inchikey_mappings)

        yy, yn, ny, nn = 0, 0, 0, 0

        for k in S:
            if d[k[0]] == d[k[1]]:
                yy += 1
            else:
                yn += 1
        for k in D:
            if d[k[0]] == d[k[1]]:
                ny += 1
            else:
                nn += 1

        M = np.array([[yy, yn], [ny, nn]])

        odds, pval = fisher_exact(M, alternative="greater")

        sns.set_style("white")
        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)
        plt.bar([1, 2], [M[0, 0] / (M[1, 0] + M[0, 0]) * 100, M[0, 1] /
                         (M[1, 1] + M[0, 1]) * 100], color=[self.color, "white"], edgecolor=self.color, lw=2)
        plt.xticks([1, 2], ["Same", "Different"])
        for h in np.arange(10, 100, 10):
            plt.plot([-1, 3], [h, h], linestyle='--', color=self.color, lw=0.3)
        plt.ylim((0, 100))
        plt.xlim((0.5, 2.5))
        plt.ylabel("% in same cluster")

        plt.title("Odds: %.2f, P-val: %.2g" % (odds, pval))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.tight_layout()
        plt.savefig("%s/%s_%s_ft_validation.png" %
                    (self.plot_path, prefix, label_type))

        with open("%s/%s_%s_ft_validation.tsv" % (self.plot_path, prefix, label_type), "w") as f:
            S = "%s\t%d\n" % ("yy", yy)
            S += "%s\t%d\n" % ("yn", yn)
            S += "%s\t%d\n" % ("ny", ny)
            S += "%s\t%d\n" % ("nn", nn)
            S += "odds\t%.2f\n" % (odds)
            S += "pval\t%.2g\n" % (pval)
            f.write(S)

        return odds, pval

    def vector_validation(self, inchikey_vec, vector_type, prefix="moa", distance="cosine", inchikey_mappings=None):

        S, D, d = self._for_the_validation(
            inchikey_vec, prefix, inchikey_mappings)

        if distance == "euclidean":
            distance_metric = euclidean
        elif distance == "cosine":
            distance_metric = cosine
        else:
            sys.exit("Unrecognized distance %s" % distance)

        S = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in S]))
        D = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in D]))

        # This exception prevents this eeror in python3 on our pytest:
        # ValueError: zero-size array to reduction operation maximum which has
        # no identity
        try:
            ks = ks_2samp(S, D)
        except ValueError:
            ks = [0.0, 0.0]
            pass

        N = len(d)

        d = None

        # Distance plot

        sns.set_style("white")

        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)

        cS = np.cumsum(S)
        cS = cS / np.max(cS)

        cD = np.cumsum(D)
        cD = cD / np.max(cD)

        plt.plot(D, cD, color=self.color, linestyle="--")
        plt.plot(S, cS, color=self.color, linestyle="-", lw=2)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)

        plt.ylim(0, 1)
        plt.xlim(np.min([np.min(S), np.min(D)]),
                 np.max([np.max(S), np.max(D)]))
        plt.xlabel("Distance")
        plt.ylabel("Cumulative proportion")

        plt.title("D: %.2f, P-val: %.2g, N: %d" % (ks[0], ks[1], N))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/%s_%s_ks_validation.png" %
                    (self.plot_path, prefix, vector_type))

        with open("%s/%s_%s_ks_validation_D.tsv" % (self.plot_path, prefix, vector_type), "w") as f:
            for i in range(len(D)):
                f.write("%f\t%f\n" % (D[i], cD[i]))

        with open("%s/%s_%s_ks_validation_S.tsv" % (self.plot_path, prefix, vector_type), "w") as f:
            for i in range(len(S)):
                f.write("%f\t%f\n" % (S[i], cS[i]))

        # ROC curve

        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)

        Scores = np.concatenate((-S, -D))
        Truth = np.array(len(S) * [1] + len(D) * [0])

        auc_score = roc_auc_score(Truth, Scores)

        fpr, tpr, thr = roc_curve(Truth, Scores)

        plt.plot([0, 1], [0, 1], color=self.color, linestyle="--")
        plt.plot(fpr, tpr, color=self.color, linestyle="-", lw=2)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)

        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.title("AUC: %.2f, N: %d" % (auc_score, N))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/%s_%s_auc_validation.png" %
                    (self.plot_path, prefix, vector_type))

        with open("%s/%s_%s_auc_validation.tsv" % (self.plot_path, prefix, vector_type), "w") as f:
            for i in range(len(fpr)):
                f.write("%f\t%f\n" % (fpr[i], tpr[i]))

        return ks, auc_score

    # Matrix plot

    def matrix_plot(self, data_path):

        sns.set_style("white")

        with h5py.File(data_path) as hf:
            Mols = len(hf["keys"])
            Vars = len(hf["V"][0])

        plt.figure(figsize=(4, 4), dpi=300)

        fig = plt.subplot(111)

        ax = plt.gca()

        xmax = 3
        ymax = 6

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        for v1 in range(1, xmax):
            plt.plot([v1, v1], [0, ymax], "-.", lw=0.5, color=self.color)
        for h1 in range(1, ymax):
            plt.plot([0, xmax], [h1, h1], "-.", lw=0.5, color=self.color)

        ax.add_patch(patches.Rectangle((0, 0), math.log10(
            Vars), math.log10(Mols), color=self.color))

        plt.yticks([t for t in range(7)])
        plt.xticks([t for t in range(4)])

        ax.set_xlabel("Latent variables (log10)")
        ax.set_ylabel("Molecules (log10)")

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        fig.patch.set_facecolor("white")

        plt.savefig("%s/matrix_plot.png" % self.plot_path)

    # Projection plot

    def projection_plot(self, Proj, bw=None, levels=5, dev=None, s=None, transparency=0.5):

        if dev:
            noise_x = np.random.normal(0, dev, Proj.shape[0])
            noise_y = np.random.normal(0, dev, Proj.shape[0])

        gray = self._rgb2hex(220, 218, 219)

        if not s:
            s = np.max([0.3, -4e-6 * Proj.shape[0] + 4])

        fig = plt.figure(figsize=(5, 5), dpi=600)
        ax = fig.add_subplot(111)

        ax.set_facecolor(self.color)
        if dev:
            X = Proj[:, 0] + noise_x
            Y = Proj[:, 1] + noise_y
        else:
            X = Proj[:, 0]
            Y = Proj[:, 1]
        ax.scatter(X, Y, alpha=transparency, s=s, color="white")
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.savefig("%s/largevis_scatter.png" % self.plot_path)

        def make_cmap(colors, position=None, bit=False):
            bit_rgb = np.linspace(0, 1, 256)
            if position is None:
                position = np.linspace(0, 1, len(colors))
            else:
                if len(position) != len(colors):
                    sys.exit("position length must be the same as colors")
                elif position[0] != 0 or position[-1] != 1:
                    sys.exit("position must start with 0 and end with 1")
            if bit:
                for i in range(len(colors)):
                    colors[i] = (bit_rgb[colors[i][0]],
                                 bit_rgb[colors[i][1]],
                                 bit_rgb[colors[i][2]])
            cdict = {'red': [], 'green': [], 'blue': []}
            for pos, color in zip(position, colors):
                cdict['red'].append((pos, color[0], color[0]))
                cdict['green'].append((pos, color[1], color[1]))
                cdict['blue'].append((pos, color[2], color[2]))

            cmap = mpl.colors.LinearSegmentedColormap(
                'my_colormap', cdict, 256)
            return cmap

        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return tuple(int(value[i:i + int(lv / 3)], 16) for i in range(0, lv, int(lv / 3)))

        def get_cmap(color):
            colors = [hex_to_rgb(c) for c in [color, gray]]
            return make_cmap(colors, bit=True)

        sns.set_style("white")

        cmap = get_cmap(self.color)

        xmin, xmax = np.min(Proj[:, 0]), np.max(Proj[:, 0])
        ymin, ymax = np.min(Proj[:, 1]), np.max(Proj[:, 1])

        margin = 0.1

        xran = xmax - xmin
        xmin = xmin - margin * xran
        xmax = xmax + margin * xran
        yran = ymax - ymin
        ymin = ymin - margin * yran
        ymax = ymax + margin * yran

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        Proj = np.vstack({tuple(row) for row in Proj})
        values = np.vstack([Proj[:, 0], Proj[:, 1]])
        kernel = gaussian_kde(values, bw_method=bw)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Plot
        fig = plt.figure(figsize=(5, 5), dpi=600)

        ax = fig.add_subplot(111)

        cut = np.percentile(f, 95)

        f[f > cut] = cut
        levels = np.linspace(0, cut, num=levels)
        f[f == np.min(f)] = 0

        ax.grid(False)
        plt.contourf(xx, yy, f, list(levels) +
                     [levels[-1] + 5 * (levels[-1] - levels[-2])], cmap=cmap)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.axes.spines["bottom"].set_color(self.color)
        ax.axes.spines["top"].set_color(self.color)
        ax.axes.spines["right"].set_color(self.color)
        ax.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/largevis.png" % self.plot_path,
                    bbox_inches="tight", pad_inches=0)

        xlim = ax.axes.get_xlim()
        ylim = ax.axes.get_ylim()

        return xlim, ylim

    def sign2_feature_distribution_plot(self, sign2):
        if sign2.shape[0] > 10000:
            keys = np.random.choice(sign2.keys, 10000, replace=False)
            matrix = sign2.get_vectors(keys)[1]
        else:
            matrix = sign2[:]
        df = pd.DataFrame(matrix).melt()

        coord = self.dataset_code
        fig = plt.figure(figsize=(10, 3), dpi=100)
        ax = fig.add_subplot(111)
        sns.pointplot(x='variable', y='value', data=df,
                      ax=ax, ci='sd', join=False, markers='.',
                      color=self._coord_color(coord))
        ax.set_ylim(-1, 1)
        ax.set_xlim(-2, 130)
        ax.set_xticks([])
        ax.set_xlabel('')
        min_mean = min(np.mean(matrix, axis=0))
        max_mean = max(np.mean(matrix, axis=0))
        ax.fill_between([-2, 130], [max_mean, max_mean], [min_mean, min_mean],
                        facecolor=self._coord_color(coord), alpha=0.4,
                        zorder=0)
        sns.despine(bottom=True)
        filename = os.path.join(self.plot_path, "feat_distrib_%s.png" % coord)
        plt.savefig(filename, dpi=100)
        plt.close()

    @skip_on_exception
    def sign2_prediction_plot(self, y_true, y_pred, predictor_name):

        coord = self.dataset_code
        self.__log.info("sign2 %s predicted vs. actual %s",
                        coord, predictor_name)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        # we are assuming a dimension of 128 components
        f, axarr = plt.subplots(
            8, 16, sharex=True, sharey=True, figsize=(60, 30))
        for comp, ax in zip(range(128), axarr.flatten()):
            rsquare = r2_score(y_true[:, comp], y_pred[:, comp])
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                y_true[:, comp], y_pred[:, comp])
            # for plotting we subsample randomly
            nr_samples = len(y_true[:, comp])
            if nr_samples > 10000:
                mask = np.random.choice(
                    range(nr_samples), 10000, replace=False)
                x = y_true[:, comp][mask]
                y = y_pred[:, comp][mask]
            else:
                x = y_true[:, comp]
                y = y_pred[:, comp]
            ax.scatter(x, y, color=self._coord_color(coord), marker='.',
                       alpha=0.3, zorder=2)
            # add lines
            ax.plot([min_val, max_val],
                    [min_val * slope + intercept, max_val * slope + intercept],
                    color=self._coord_color(coord))
            ax.plot([min_val, max_val], [min_val, max_val], '--',
                    color=self._coord_color(coord))
            ax.axhline(0, min_val, max_val, ls='--',
                       lw=0.5, color='grey', zorder=1)
            ax.axvline(0, min_val, max_val, ls='--',
                       lw=0.5, color='grey', zorder=1)
            # fix limits considering all components dynamic range
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            # add R square stat
            cmap = plt.cm.get_cmap('jet_r')
            txt = ax.text(0.05, 0.85, "$R^2$: {:.2f}".format(rsquare),
                          transform=ax.transAxes,
                          size=30, color=cmap(rsquare / 2))
            txt.set_path_effects([path_effects.Stroke(
                linewidth=1, foreground='black'), path_effects.Normal()])
            # visualize linear regression equation
            ax.text(0.2, 0.05,
                    "$y = {:.2f}x {:+.2f}$".format(slope, intercept),
                    transform=ax.transAxes,
                    size=20, color='k')
            """
            # following would make one plot for each component along with
            # distributions (takes too long)
            g = sns.JointGrid(x=y_true[:, comp], y=y_pred[:, comp], space=0)
            g = g.plot_joint(sns.regplot, color="r")
            g = g.plot_joint(sns.kdeplot, cmap="Greys")
            g = g.plot_marginals(sns.kdeplot, color="r", shade=True)
            rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
            g = g.annotate(rsquare, template="{stat}: {val:.2f}",
                           stat="$R^2$", loc="upper left", fontsize=12)
            plt.savefig('pred_vs_true_%s_%s.png' % (coord, comp), dpi=100)
            """
        f.tight_layout()
        filename = os.path.join(
            self.plot_path, "pred_vs_true_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

        ####
        # relationship between predicted and observed distances
        ####
        self.__log.info("sign2 %s DISTANCES predicted vs. actual %s",
                        coord, predictor_name)

        # for plotting we subsample randomly
        nr_samples = len(y_true)
        if nr_samples > 1000:
            mask = np.random.choice(
                range(nr_samples), 1000, replace=False)
            x = cosine_distances(y_true[mask])[np.tril_indices(1000, -1)]
            y = cosine_distances(y_pred[mask])[np.tril_indices(1000, -1)]
        else:
            x = cosine_distances(y_true)[np.tril_indices(nr_samples, -1)]
            y = cosine_distances(y_pred)[np.tril_indices(nr_samples, -1)]
        # stats
        pearson = stats.pearsonr(x, y)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, y)
        # plot
        f, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x, y, color=self._coord_color(coord), marker='.',
                   alpha=0.2, zorder=2)
        # add lines
        min_val = 0.0
        max_val = 1.0
        ax.plot([min_val, max_val],
                [min_val * slope + intercept, max_val * slope + intercept],
                color=self._coord_color(coord))
        ax.plot([min_val, max_val], [min_val, max_val], '--',
                color=self._coord_color(coord))
        ax.axhline(0, min_val, max_val, ls='--',
                   lw=0.5, color='grey', zorder=1)
        ax.axvline(0, min_val, max_val, ls='--',
                   lw=0.5, color='grey', zorder=1)
        # fix limits of distance metric
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # set labels
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Distances")
        # add R square stat
        cmap = plt.cm.get_cmap('jet_r')
        txt = ax.text(0.05, 0.85, "$\\rho$: {:.2f}".format(pearson),
                      transform=ax.transAxes,
                      size=30, color=cmap(pearson / 2))
        txt.set_path_effects([path_effects.Stroke(
            linewidth=1, foreground='black'), path_effects.Normal()])
        # visualize linear regression equation
        ax.text(0.2, 0.05,
                "$y = {:.2f}x {:+.2f}$".format(slope, intercept),
                transform=ax.transAxes,
                size=20, color='k')

        f.tight_layout()
        filename = os.path.join(
            self.plot_path, "DISTANCES_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

        self.__log.info("sign2 %s DISTANCES KDE predicted vs. actual %s",
                        coord, predictor_name)
        if len(x) > 1000:
            mask = np.random.choice(range(len(x)), 1000, replace=False)
            x = x[mask]
            y = y[mask]
        sns.jointplot(x, y, kind="kde", color=self._coord_color(coord),
                      height=7, xlim=(0, 1), ylim=(0, 1))
        filename = os.path.join(
            self.plot_path, "DISTANCES_kde_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

    def sign2_grid_search_plot(self, grid_root=None):
        dir_names = [name for name in os.listdir(
            grid_root) if os.path.isdir(os.path.join(grid_root, name))]
        for dir_name in dir_names:
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if os.path.isfile(tmpdf_file):
                break
        cols = list(pd.read_pickle(tmpdf_file).columns)
        df = pd.DataFrame(columns=set(cols) | set(params.keys()))
        for dir_name in dir_names:
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if not os.path.isfile(tmpdf_file):
                print("File not found: %s", tmpdf_file)
                continue
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf = pd.read_pickle(tmpdf_file)
            for k, v in params.items():
                tmpdf[k] = pd.Series([v] * len(tmpdf))
            df = df.append(tmpdf, ignore_index=True)

        df['layer_size'] = df['layer_size'].astype(int)
        df['adanet_iterations'] = df['adanet_iterations'].astype(int)
        df['adanet_lambda'] = df['adanet_lambda'].astype(float)

        for metric in ['pearson_avg', 'nr_variables']:
            ada_df = df[df.algo == 'AdaNet']
            hue_order = sorted(list(ada_df.subnetwork_generator.unique()))
            g = sns.relplot(y=metric, style="dataset", hue='subnetwork_generator',
                            x='adanet_lambda', col='adanet_iterations',
                            hue_order=hue_order,
                            kind='scatter', data=ada_df)
            # linreg
            linreg_train = df[(df.algo != 'AdaNet') & (
                df.dataset == 'train')].iloc[0][metric]
            linreg_test = df[(df.algo != 'AdaNet') & (
                df.dataset == 'test')].iloc[0][metric]
            for ax in g.axes.flatten():
                ax.axhline(linreg_train, ls='--',
                           lw=0.5, color='grey', zorder=1)
                ax.axhline(linreg_test,
                           lw=0.5, color='grey', zorder=1)

            filename = os.path.join(
                self.plot_path, "sign2_%s_grid_search_%s.png" % (self.dataset_code, metric))
            plt.savefig(filename, dpi=100)
            plt.close()
