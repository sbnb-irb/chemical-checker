"""Plot information on multiple Chemical Checker datasets."""
import os
import math
import h5py
import json
import pickle
import itertools
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from scipy import interpolate
from scipy import stats
from functools import partial
from scipy.stats import gaussian_kde
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from matplotlib.patches import Polygon
from sklearn.preprocessing import robust_scale
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.manifold import MDS
from scipy.spatial.distance import cosine, euclidean
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection

from chemicalchecker.util.parser import Converter
from chemicalchecker.util import logged
from chemicalchecker.util.decomposition import dataset_correlation
from chemicalchecker.util.plot.diagnosticsplot import DiagnosisPlot


@logged
class MultiPlot():
    """MultiPlot class.

    Produce Chemical Checker plots using multiple datasets.
    """

    def __init__(self, chemchecker, plot_path, limit_dataset=None, svg=False,
                 dpi=200, grey=False, style=None):
        """Initialize a MultiPlot instance.

        Produce plots integrating data from multiple datasets.

        Args:
            chemchecker (str): A Chemical Checker instance.
            plot_path (str): Destination folder for plot images.
        """
        if not os.path.isdir(plot_path):
            raise Exception("Folder to save plots does not exist")
        self.__log.debug('Plots will be saved to %s', plot_path)
        self.plot_path = plot_path
        self.cc = chemchecker
        if not limit_dataset:
            self.datasets = list(self.cc.datasets)
        else:
            if not isinstance(limit_dataset, list):
                self.datasets = list(limit_dataset)
            else:
                self.datasets = limit_dataset
        self.svg = svg
        self.dpi = dpi
        self.grey = grey
        if style is None:
            self.style = ('ticks', {
                'font.family': ' sans-serif',
                'font.serif': ['Arial'],
                'font.size': 16,
                'axes.grid': True})
        else:
            self.style = style
        sns.set_style(*self.style)

    def _rgb2hex(self, r, g, b):
        return '#%02x%02x%02x' % (r, g, b)

    def cc_palette(self, coords):
        """Return a list of colors a.k.a. a palette."""
        def rgb2hex(r, g, b):
            return '#%02x%02x%02x' % (r, g, b)

        colors = list()
        for coord in coords:
            if "A" in coord:
                colors.append(rgb2hex(250, 100, 80))
            elif "B" in coord:
                colors.append(rgb2hex(200, 100, 225))
            elif "C" in coord:
                colors.append(rgb2hex(80,  120, 220))
            elif "D" in coord:
                colors.append(rgb2hex(120, 180, 60))
            elif "E" in coord:
                colors.append(rgb2hex(250, 150, 50))
        return colors

    def cc_colors(self, coord, lighness=0):
        colors = {
            'A': ['#EA5A49', '#EE7B6D', '#F7BDB6'],
            'B': ['#B16BA8', '#C189B9', '#D0A6CB'],
            'C': ['#5A72B5', '#7B8EC4', '#9CAAD3'],
            'D': ['#7CAF2A', '#96BF55', '#B0CF7F'],
            'E': ['#F39426', '#F5A951', '#F8BF7D'],
            'G': ['#333333', '#666666', '#999999']}
        if not self.grey:
            return colors[coord[:1]][lighness]
        else:
            return colors['G'][lighness]

    def cmap_discretize(cmap, N):
        """Return a discrete colormap from the continuous colormap cmap.

            cmap: colormap instance, eg. cm.jet.
            N: number of colors.

        Example
            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
            imshow(x, cmap=djet)
        """

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
        colors_rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N + 1)
        cdict = {}
        for ki, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [(indices[i], colors_rgba[i - 1, ki],
                           colors_rgba[i, ki]) for i in range(N + 1)]
        # Return colormap object.
        return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

    def sign_adanet_stats(self, ctype, metric=None, compare=None):
        # read stats fields
        sign = self.cc.get_signature(ctype, 'full', 'E5.001')
        stat_file = os.path.join(
            sign.model_path, 'adanet_eval', 'stats_eval.pkl')
        df = pd.read_pickle(stat_file)
        # merge all stats to pandas
        df = pd.DataFrame(columns=['coordinate'] + list(df.columns))
        for ds in tqdm(self.datasets):
            sign = self.cc.get_signature(ctype, 'full', ds)
            stat_file = os.path.join(
                sign.model_path, 'adanet_eval', 'stats_eval.pkl')
            if not os.path.isfile(stat_file):
                continue
            tmpdf = pd.read_pickle(stat_file)
            tmpdf['coordinate'] = ds
            df = pd.concat([ df, tmpdf], ignore_index=True)
        df = df.infer_objects()

        outfile_csv = os.path.join(self.plot_path, 'sign2_adanet_stats.csv')
        df.to_csv(outfile_csv)
        outfile_pkl = os.path.join(self.plot_path, 'sign2_adanet_stats.pkl')
        df.to_pickle(outfile_pkl)

        if compare:
            cdf = pd.read_pickle(compare)
            cdf = cdf[cdf.algo == 'AdaNet'].copy()
            cdf['algo'] = cdf.algo.apply(lambda x: x + '_STACK')
            df = pd.concat([ df, cdf], ignore_index=True)

        if metric:
            all_metrics = [metric]
        else:
            all_metrics = ['mse', 'r2', 'explained_variance', 'pearson_std',
                           'pearson_avg', 'time', 'nn_layers', 'nr_variables']
        for metric in all_metrics:
            # sns.set_style("whitegrid")
            g = sns.catplot(data=df, kind='point', x='dataset', y=metric,
                            hue="algo", col="coordinate", col_wrap=5,
                            col_order=self.datasets,
                            aspect=.8, height=3, dodge=True,
                            order=['train', 'test', 'validation'],
                            palette=['darkgreen', 'orange', 'darkgrey'])
            if metric == 'r2':
                for ax in g.axes.flatten():
                    ax.set_ylim(0, 1)
            if metric == 'mse':
                for ax in g.axes.flatten():
                    ax.set_ylim(0, 0.02)
            if metric == 'explained_variance':
                for ax in g.axes.flatten():
                    ax.set_ylim(0, 1)

            if compare:
                metric += '_CMP'
            outfile = os.path.join(
                self.plot_path, 'sign2_adanet_stats_%s.png' % metric)
            plt.savefig(outfile, dpi=self.dpi)
            plt.close('all')

    def sign2_node2vec_stats(self):
        """Plot the stats for sign2."""

        # plot selected stats
        stats = [
            "nodes",
            "edges",
            #"zeroNodes",
            "zeroInNodes",
            #"zeroOutNodes",
            #"nonZIODegNodes",
            "Connected Components",
            "Degree",
            "Weights",
            "AUC-ROC",
            "MCC",
            "Sign Range"]
        # the following are aggregated
        conncompo = [
            "SccSz",
            "WccSz",
        ]
        degrees = [
            #"Degree_min",
            "Degree_25",
            "Degree_50",
            "Degree_75",
            #"Degree_max"
        ]
        weights = [
            #"Weight_min",
            "Weight_25",
            "Weight_50",
            "Weight_75",
            #"Weight_max"
        ]

        # move stats to pandas
        df = pd.DataFrame(columns=['dataset'] + stats)
        for ds in tqdm(self.datasets):
            # get sign2 and stats file
            sign2 = self.cc.get_signature('sign2', 'reference', ds)
            graph_file = os.path.join(sign2.stats_path, "graph_stats.json")
            if not os.path.isfile(graph_file):
                self.__log.warn('Graph stats %s not found', graph_file)
                continue
            graph_stat = json.load(open(graph_file, 'r'))
            linkpred_file = os.path.join(sign2.stats_path, "linkpred.json")
            skip_linkpred = False
            if not os.path.isfile(linkpred_file):
                self.__log.warn('Node2vec stats %s not found', linkpred_file)
                skip_linkpred = True
                pass
            if not skip_linkpred:
                liknpred_perf = json.load(open(linkpred_file, 'r'))
                liknpred_perf = {k: float(v) for k, v in liknpred_perf.items()}
            # prepare row
            for deg in degrees:
                row = dict()
                row.update(graph_stat)
                if not skip_linkpred:
                    row.update(liknpred_perf)
                row.update({"dataset": ds})
                row.update({"Degree": graph_stat[deg]})
                df.loc[len(df)] = pd.Series(row)
            for conn in conncompo:
                row = dict()
                row.update(graph_stat)
                if not skip_linkpred:
                    row.update(liknpred_perf)
                row.update({"dataset": ds})
                row.update({"Connected Components": graph_stat[conn]})
                df.loc[len(df)] = pd.Series(row)
            for wei in weights:
                row = dict()
                row.update(graph_stat)
                if not skip_linkpred:
                    row.update(liknpred_perf)
                row.update({"dataset": ds})
                row.update({"Weights": graph_stat[wei]})
                df.loc[len(df)] = pd.Series(row)
            maxss = list()
            minss = list()
            for s in sign2.chunker(size=10000):
                curr = sign2[s]
                maxss.append(np.percentile(curr, 99))
                minss.append(np.percentile(curr, 1))
            row = {"dataset": ds, "Sign Range": np.mean(maxss)}
            df.loc[len(df)] = pd.Series(row)
            row = {"dataset": ds, "Sign Range": np.mean(minss)}
            df.loc[len(df)] = pd.Series(row)

        df = df.infer_objects()
        sns.set(style="ticks")
        sns.set_context("talk", font_scale=1.)
        g = sns.PairGrid(df.sort_values("dataset", ascending=True),
                         x_vars=stats, y_vars=["dataset"],
                         height=10, aspect=.3)
        g.map(sns.stripplot, size=10, dodge=False, jitter=False,  # marker="|",
              palette=self.cc_palette(self.datasets),
              orient="h", linewidth=1, edgecolor="w")

        for ax in g.axes.flat:
            # Make the grid horizontal instead of vertical
            ax.xaxis.grid(True, color='#e3e3e3')
            ax.yaxis.grid(True)

        g.axes.flat[0].set_xscale("log")
        g.axes.flat[0].set_xlim(1e3, 3 * 1e6)
        g.axes.flat[0].set_xlabel("Nodes")
        g.axes.flat[1].set_xscale("log")
        g.axes.flat[1].set_xlim(1e4, 1e8)
        g.axes.flat[1].set_xlabel("Edges")
        g.axes.flat[2].set_xlim(0, 5000)
        g.axes.flat[2].set_xlabel("0 In Nodes")
        g.axes.flat[3].set_xlim(0, 1.0)
        g.axes.flat[4].set_xscale("log")
        g.axes.flat[4].set_xlim(10, 1e3)
        g.axes.flat[4].set_xlabel("Degree %tiles")
        g.axes.flat[5].set_xlim(0, 1)
        g.axes.flat[6].set_xlim(0.9, 1)
        g.axes.flat[7].set_xlim(0.5, 1)
        g.axes.flat[8].set_xlim(-1, 1)
        # g.axes.flat[-1].set_xlim(1e1,1e3)
        sns.despine(left=True, bottom=True)

        outfile = os.path.join(self.plot_path, 'sign2_node2vec_stats.png')
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign_feature_distribution_plot(self, cctype, molset, block_size=1000,
                                       block_nr=10, sort=False):
        sample_size = block_size * block_nr
        fig, axes = plt.subplots(25, 1, sharey=True, sharex=True,
                                 figsize=(10, 40), dpi=self.dpi)
        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):
            sign = self.cc.get_signature(cctype, molset, ds)
            if not os.path.isfile(sign.data_path):
                continue
            if sign.shape[0] > sample_size:
                blocks = np.random.choice(
                    int(np.ceil(sample_size / block_size)) + 1,
                    block_nr, replace=False)
                block_mat = list()
                for block in tqdm(blocks):
                    chunk = slice(block * block_size,
                                  (block * block_size) + block_size)
                    block_mat.append(sign[chunk])
                matrix = np.vstack(block_mat)
            else:
                matrix = sign[:]
            df = pd.DataFrame(matrix).melt()
            all_df = df.copy()
            all_df['variable'] = 130
            df = pd.concat([ df, all_df], ignore_index=True)
            if not sort:
                order = [130, -1] + range(matrix.shape[1])
            else:
                order = [130, -1] + \
                    list(np.argsort(np.mean(matrix, axis=0))[::-1])
            sns.pointplot(x='variable', y='value', data=df, order=order,
                          ax=ax, ci='sd', join=False, markers='.',
                          color=self.cc_palette([ds])[0])
            ax.set_ylim(-1, 1)
            ax.set_xlim(-2, 130)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel(ds)
            min_mean = np.min(np.mean(matrix, axis=0))
            max_mean = np.max(np.mean(matrix, axis=0))
            ax.fill_between([-2, 130], [max_mean, max_mean],
                            [min_mean, min_mean],
                            facecolor=self.cc_palette([ds])[0], alpha=0.3,
                            zorder=0)
            max_std = max(np.std(matrix, axis=0))
            ax.fill_between([-2, 130],
                            [max_mean + max_std, max_mean + max_std],
                            [min_mean - max_std, min_mean - max_std],
                            facecolor=self.cc_palette([ds])[0], alpha=0.2,
                            zorder=0)
            sns.despine(bottom=True)
        plt.tight_layout()
        if not sort:
            filename = os.path.join(
                self.plot_path,
                "%s_%s_feat_distrib.png" % (cctype, molset))
        else:
            filename = os.path.join(
                self.plot_path,
                "%s_%s_feat_distrib_sort.png" % (cctype, molset))
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def plot_adanet_subnetwork_layer_size(self, shapes=None, func=None):

        if not shapes:
            shapes = list()
            for ds in self.cc.datasets:
                sign1 = self.cc.get_signature('sign1', 'reference', ds)
                x, y = sign1.shape
                shapes.append((ds, x, y))

        def layer_size(nr_samples, nr_features, nr_out=128, s_fact=7.):
            heu_layer_size = (
                1 / s_fact) * (np.sqrt(nr_samples) / .3 + ((nr_features + nr_out) / 5.))
            heu_layer_size = np.power(2, np.ceil(np.log2(heu_layer_size)))
            heu_layer_size = np.maximum(heu_layer_size, 32)
            return heu_layer_size

        if not func:
            func = layer_size

        x = np.logspace(2, 6, 500)
        y = np.linspace(5, 5000, 500)
        X, Y = np.meshgrid(x, y)  # grid of point
        Z = func(X, Y)  # evaluation of the function on the grid

        # sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5), dpi=self.dpi)
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=[2**i for i in range(5, 11)], ncolors=256)
        # drawing the function
        im = ax.pcolormesh(X, Y, Z, norm=norm, cmap=plt.cm.Blues)
        plt.xscale('log')
        # adding the Contour lines with labels
        # cset = ax.contour(Z, [2**i for i in range(2, 11)],linewidths = 2, cmap = plt.cm.Set2)
        # plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
        plt.colorbar(im, label='Neurons')  # adding the colobar on the right
        plt.ylim(5, 5000)
        ax.set_xlabel("Molecules")
        ax.set_ylabel("Features")
        plt.tight_layout()

        for ds, x, y in shapes:
            plt.scatter(x, y, color=self.cc_palette([ds])[0], alpha=.3)
            plt.text(x, y, "%s" % (ds[:2]),
                     ha="center", va="center",
                     bbox={"boxstyle": "circle", "color": self.cc_palette([ds])[
                         0]},
                     color='k', fontsize=10)

        filename = os.path.join(self.plot_path, "layer_size.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def sign2_grid_search_plot(self, grid_postfix=None):
        grid_roots = list()
        for ds in self.cc.datasets:
            sign2 = self.cc.get_signature('sign2', 'reference', ds)
            grid_roots.append(os.path.join(sign2.model_path,
                                           'grid_search_%s' % grid_postfix))
        file_names = list()
        for grid_root in grid_roots:
            file_names.extend([os.path.join(grid_root, name, 'stats.pkl') for name in os.listdir(
                grid_root) if os.path.isfile(os.path.join(grid_root, name, 'stats.pkl'))])

        cols = list(pd.read_pickle(file_names[0]).columns)
        params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                  for n in file_names[0].split('/')[-2].split("-")}
        df = pd.DataFrame(columns=set(cols) | set(params.keys()))
        for tmpdf_file in file_names:
            tmpdf = pd.read_pickle(tmpdf_file)
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in tmpdf_file.split('/')[-2].split("-")}
            for k, v in params.items():
                tmpdf[k] = pd.Series([v] * len(tmpdf))
            coordinate = tmpdf_file.split('/')[-6]
            tmpdf['coordinate'] = pd.Series([coordinate] * len(tmpdf))
            if 'Ext' in params["subnetwork_generator"]:
                tmpdf = tmpdf[tmpdf.algo == 'AdaNet']
            else:
                tmpdf["subnetwork_generator"] = tmpdf.algo.map(
                    {"AdaNet": "StackDNNGenerator", "LinearRegression": "LinearRegression"})
            df = pd.concat([ df, tmpdf], ignore_index=True)

        # df['layer_size'] = df['layer_size'].astype(int)
        # df['adanet_iterations'] = df['adanet_iterations'].astype(int)
        # df['adanet_lambda'] = df['adanet_lambda'].astype(float)
        df = df.infer_objects()
        sns.set_context("talk")
        netdf = pd.DataFrame(columns=list(df.columns) + ['layer', 'neurons'])
        for index, row in df.iterrows():
            for layer, size in enumerate(row.architecture[:-1]):
                new_row = row.to_dict()
                new_row['layer'] = layer + 1
                new_row['neurons'] = size
                netdf.loc[len(netdf)] = pd.Series(new_row)

        sns.set_context("notebook")
        # sns.set_style("whitegrid")
        hue_order = ["StackDNNGenerator", "ExtendDNNGenerator"]
        g = sns.catplot(data=netdf, kind='bar', x='layer', y='neurons',
                        hue="subnetwork_generator", col="coordinate", col_wrap=5,
                        col_order=self.datasets, hue_order=hue_order,
                        aspect=1.2, height=3, dodge=True,
                        palette=['forestgreen', 'orange'])
        for ax in g.axes.flatten():
            ax.set_yscale('log', basey=2)
            ax.set_title("")
        filename = os.path.join(
            self.plot_path, "sign2_%s_grid_search_NN.png" % (grid_postfix))
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

        hue_order = ["StackDNNGenerator", "ExtendDNNGenerator"]
        g = sns.catplot(data=netdf[netdf.subnetwork_generator == 'StackDNNGenerator'], kind='bar', x='layer', y='neurons',
                        hue="subnetwork_generator", col="coordinate", col_wrap=5,
                        col_order=self.datasets, hue_order=[
                            "StackDNNGenerator"],
                        aspect=1.2, height=3, dodge=True,
                        palette=['forestgreen', 'orange'])
        for ax in g.axes.flatten():
            ax.set_yscale('log', basey=2)
            ax.set_title("")
        filename = os.path.join(
            self.plot_path, "sign2_%s_grid_search_NN_stackonly.png" % (grid_postfix))
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

        for metric in ['pearson_avg', 'time', 'r2', 'pearson_std', 'explained_variance']:
            # sns.set_style("whitegrid")
            hue_order = ["StackDNNGenerator",
                         "ExtendDNNGenerator", "LinearRegression"]
            if metric == 'time':
                sharey = False
            else:
                sharey = True
            g = sns.catplot(data=df, kind='point', x='dataset', y=metric,
                            hue="subnetwork_generator", col="coordinate", col_wrap=5,
                            col_order=self.datasets, hue_order=hue_order, sharey=sharey,
                            order=['train', 'test', 'validation'], aspect=1.2, height=3,
                            palette=['forestgreen', 'orange', 'darkgrey'])

            for ax in g.axes.flatten():
                if metric == 'pearson_avg':
                    ax.set_ylim(0.5, 1)
                ax.set_title("")
            filename = os.path.join(
                self.plot_path, "sign2_%s_grid_search_%s.png" % (grid_postfix, metric))
            plt.savefig(filename, dpi=self.dpi)
            plt.close()

            g = sns.catplot(data=df[df.subnetwork_generator != 'ExtendDNNGenerator'], kind='point', x='dataset', y=metric,
                            hue="subnetwork_generator", col="coordinate", col_wrap=5,
                            col_order=self.datasets, hue_order=[
                                "StackDNNGenerator", "LinearRegression"],
                            aspect=1.2, height=3, dodge=True, sharey=sharey,
                            order=['train', 'test', 'validation'],
                            palette=['forestgreen', 'darkgrey'])

            for ax in g.axes.flatten():
                if metric == 'pearson_avg':
                    ax.set_ylim(0.5, 1)
                ax.set_title("")
            filename = os.path.join(
                self.plot_path, "sign2_%s_grid_search_%s_stackonly.png" % (grid_postfix, metric))
            plt.savefig(filename, dpi=self.dpi)
            plt.close()

    def sign2_grid_search_node2vec_plot(self, grid_postfix=None):
        grid_roots = list()
        for ds in self.cc.datasets:
            sign2 = self.cc.get_signature('sign2', 'reference', ds)
            grid_roots.append(os.path.join(sign2.model_path,
                                           'grid_search_%s' % grid_postfix))
        file_names = list()
        for grid_root in grid_roots:
            file_names.extend([os.path.join(grid_root, name, 'linkpred.test.json') for name in os.listdir(
                grid_root) if os.path.isfile(os.path.join(grid_root, name, 'linkpred.test.json'))])
            file_names.extend([os.path.join(grid_root, name, 'linkpred.train.json') for name in os.listdir(
                grid_root) if os.path.isfile(os.path.join(grid_root, name, 'linkpred.train.json'))])

        cols = json.load(open(file_names[0], 'r')).keys()
        params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                  for n in file_names[0].split('/')[-2].split("-")}
        columns = list(set(cols) | set(params.keys()))
        df = pd.DataFrame(columns=columns + ['coordinate', 'dataset'])
        for tmpdf_file in file_names:
            row = json.load(open(tmpdf_file, 'r'))
            row = {k: float(v) for k, v in row.items()}
            row['coordinate'] = tmpdf_file.split('/')[-6]
            if 'train' in tmpdf_file:
                row['dataset'] = 'train'
            else:
                row['dataset'] = 'test'
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in tmpdf_file.split('/')[-2].split("-")}
            row.update(params)
            df.loc[len(df)] = pd.Series(row)
        df['d'] = df['d'].astype(int)
        df = df.infer_objects()

        sns.set_context("talk")
        # sns.set_style("ticks")
        g = sns.relplot(data=df, kind='line', x='d', y='AUC-ROC',
                        hue="coordinate", col="coordinate", col_wrap=5,
                        col_order=self.datasets, style="dataset",
                        palette=self.cc_palette(self.datasets),
                        aspect=1, height=2.475, legend=False, lw=3)
        g.fig.set_size_inches(16.5, 16.5)
        g.set_titles("")
        coords = {0: "$\\bf{A}$", 5: "$\\bf{B}$",
                  10: "$\\bf{C}$", 15: "$\\bf{D}$", 20: "$\\bf{E}$"}
        for idx, ax in enumerate(g.axes.flatten()):
            ax.set_xscale('log', basex=2)
            ax.set_xticks([2,  16, 128, 1024])
            ax.set_yticks([.8, .9, 1.0])
            ax.set_ylim([.78, 1.02])
            if not idx % 5:
                ax.set_ylabel(coords[idx])
            if idx >= 20:
                ax.set_xlabel("$\\bf{%s}$" % ((idx % 5) + 1))
            if idx == 24:
                lines = [matplotlib.lines.Line2D(
                    [0], [0], color=".15",
                    linewidth=2, linestyle=ls) for ls in ['--', '-']]
                labels = ['Train', 'Test']
                ax.legend(lines, labels, frameon=False)
        sns.despine(top=False, right=False, left=False, bottom=False)
        filename = os.path.join(
            self.plot_path, "sign2_%s_grid_search_node2vec.png" % (grid_postfix))
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def sign3_grid_search_plot(self, grid_roots):
        file_names = list()
        for grid_root in grid_roots:
            file_names.extend([os.path.join(grid_root, name, 'adanet', 'stats.pkl') for name in os.listdir(
                grid_root) if os.path.isfile(os.path.join(grid_root, name, 'adanet', 'stats.pkl'))])

        cols = list(pd.read_pickle(file_names[0]).columns)
        df = pd.DataFrame(columns=list(set(cols)) + ['subnetwork_generator'])
        for tmpdf_file in file_names:
            tmpdf = pd.read_pickle(tmpdf_file)
            coordinate = tmpdf_file.split('/')[-3]
            tmpdf['coordinate'] = pd.Series(
                [coordinate.split("_")[0]] * len(tmpdf))
            if 'STACK' in tmpdf_file:
                tmpdf = tmpdf[tmpdf.algo == 'AdaNet']
                tmpdf["subnetwork_generator"] = tmpdf.algo.map(
                    {"AdaNet": "StackDNNGenerator", "LinearRegression": "LinearRegression"})
            else:
                tmpdf["subnetwork_generator"] = tmpdf.algo.map(
                    {"AdaNet": "ExtendDNNGenerator", "LinearRegression": "LinearRegression"})
            df = pd.concat([ df, tmpdf], ignore_index=True)

        # df['layer_size'] = df['layer_size'].astype(int)
        # df['adanet_iterations'] = df['adanet_iterations'].astype(int)
        # df['adanet_lambda'] = df['adanet_lambda'].astype(float)
        df = df.infer_objects()
        sns.set_context("talk")
        netdf = pd.DataFrame(columns=list(df.columns) + ['layer', 'neurons'])
        for index, row in df.iterrows():
            for layer, size in enumerate(row.architecture[:-1]):
                new_row = row.to_dict()
                new_row['layer'] = layer + 1
                new_row['neurons'] = size
                netdf.loc[len(netdf)] = pd.Series(new_row)

        sns.set_context("notebook")
        # sns.set_style("whitegrid")
        hue_order = ["StackDNNGenerator", "ExtendDNNGenerator"]
        g = sns.catplot(data=netdf, kind='bar', x='layer', y='neurons',
                        hue="subnetwork_generator", col="coordinate", col_wrap=5,
                        col_order=self.datasets, hue_order=hue_order,
                        aspect=1.2, height=3, dodge=True,
                        palette=['forestgreen', 'orange'])
        for ax in g.axes.flatten():
            ax.set_yscale('log', basey=2)
            ax.set_title("")
        filename = os.path.join(
            self.plot_path, "sign3_crossfit_NN.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

        hue_order = ["StackDNNGenerator", "ExtendDNNGenerator"]
        g = sns.catplot(data=netdf[netdf.subnetwork_generator == 'StackDNNGenerator'], kind='bar', x='layer', y='neurons',
                        hue="subnetwork_generator", col="coordinate", col_wrap=5,
                        col_order=self.datasets, hue_order=[
                            "StackDNNGenerator"],
                        aspect=1.2, height=3, dodge=True,
                        palette=['forestgreen', 'orange'])
        for ax in g.axes.flatten():
            ax.set_yscale('log', basey=2)
            ax.set_title("")
        filename = os.path.join(
            self.plot_path, "sign3_crossfit_NN_stackonly.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

        for metric in ['pearson_avg', 'time', 'r2', 'pearson_std', 'explained_variance']:
            # sns.set_style("whitegrid")
            hue_order = ["StackDNNGenerator",
                         "ExtendDNNGenerator", "LinearRegression"]
            if metric == 'time':
                sharey = False
            else:
                sharey = True
            g = sns.catplot(data=df, kind='point', x='dataset', y=metric,
                            hue="subnetwork_generator", col="coordinate", col_wrap=5,
                            col_order=self.datasets, hue_order=hue_order, sharey=sharey,
                            order=['train', 'test', 'validation'], aspect=1.2, height=3,
                            palette=['forestgreen', 'orange', 'darkgrey'])

            for ax in g.axes.flatten():
                ax.set_title("")
            filename = os.path.join(
                self.plot_path, "sign3_crossfit_%s.png" % (metric))
            plt.savefig(filename, dpi=self.dpi)
            plt.close()

            g = sns.catplot(data=df[df.subnetwork_generator != 'ExtendDNNGenerator'], kind='point', x='dataset', y=metric,
                            hue="subnetwork_generator", col="coordinate", col_wrap=5,
                            col_order=self.datasets, hue_order=[
                                "StackDNNGenerator", "LinearRegression"],
                            aspect=1.2, height=3, dodge=True, sharey=sharey,
                            order=['train', 'test', 'validation'],
                            palette=['forestgreen', 'darkgrey'])

            for ax in g.axes.flatten():
                ax.set_title("")
            filename = os.path.join(
                self.plot_path, "sign3_crossfit_%s_stackonly.png" % (metric))
            plt.savefig(filename, dpi=self.dpi)
            plt.close()

    def sign3_all_crossfit_plot(self, crossfit_dir):
        file_names = list()
        for name in os.listdir(crossfit_dir):
            filename = os.path.join(crossfit_dir, name, 'adanet', 'stats.pkl')
            if not os.path.isfile(filename):
                print("File not found: %s", filename)
                continue
            file_names.append(filename)

        cols = list(pd.read_pickle(file_names[0]).columns)
        df = pd.DataFrame(columns=list(set(cols)) +
                          ['coordinate_from', 'coordinate_to', 'train_size'])
        #dfs = list()
        for tmpdf_file in file_names:
            tmpdf = pd.read_pickle(tmpdf_file)
            coordinate = tmpdf_file.split('/')[-3]
            tmpdf['coordinate_from'] = pd.Series(
                [coordinate.split("_")[0]] * len(tmpdf))
            tmpdf['coordinate_to'] = pd.Series(
                [coordinate.split("_")[1]] * len(tmpdf))
            # also find dataset size
            traintest_file = os.path.join(os.path.split(tmpdf_file)[
                                          0], '..', 'traintest.h5')
            with h5py.File(traintest_file, 'r') as fh:
                train_size = fh['x_train'].shape[0]
            tmpdf['train_size'] = pd.Series([train_size] * len(tmpdf))
            df = pd.concat([ df, tmpdf], ignore_index=True)
            #dfs.append(tmpdf)
        #df = pd.concat([ df, dfs], ignore_index=True)
        df = df.infer_objects()

        adanet_test = df[(df.algo == 'AdaNet') & (df.dataset == 'test')]
        adanet_train = df[(df.algo == 'AdaNet') & (df.dataset == 'train')]

        metrics = ['pearson_avg', 'time', 'r2', 'train_size',
                   'pearson_std', 'explained_variance', 'nr_variables']

        for idx, met in enumerate(metrics):
            piv = adanet_test.pivot(
                index='coordinate_from', columns='coordinate_to', values=met)
            if met == 'train_size':
                piv = np.log10(piv)
                met = 'log_train_size'
            if met == 'nr_variables':
                piv = np.log10(piv)
                met = 'log_nr_variables'
            ax = plt.axes()
            col_start = np.linspace(0, 3, len(metrics))[idx]
            col_rot = 1. / len(metrics)
            cubehelix = sns.cubehelix_palette(
                start=col_start, rot=col_rot, as_cmap=True)
            cubehelix.set_under(".9")
            cmap = self.cmap_discretize(cubehelix, 5)
            cmap.set_under(".9")
            if met == 'pearson_avg':
                sns.heatmap(piv, cmap=cmap,
                            linecolor='grey',
                            square=True, vmin=0., vmax=1., ax=ax)
            else:
                sns.heatmap(piv, cmap=cmap,
                            linecolor='grey',
                            square=True, ax=ax)
            """
            elif met in ['F1', 'AUC-ROC', 'AUC-PR']:
                sns.heatmap(piv, cmap=cmap,
                            linecolor='grey',
                            square=True, vmin=0.5, vmax=1., ax=ax)
            elif met in ['validation_neg_median', 'validation_pos_median']:
                sns.heatmap(piv, cmap=cmap_discretize(cubehelix, 10),
                            linecolor='grey',
                            square=True, vmin=-1., vmax=1., ax=ax)
            """
            for grid in range(0, 26, 5):
                ax.axhline(y=grid, color='grey', linewidth=0.5)
                ax.axvline(x=grid, color='grey', linewidth=0.5)
            ax.set_title(met)
            plt.tight_layout()
            filename = os.path.join(
                self.plot_path, "sign3_all_crossfit_train_delta_%s.png" % (met))
            plt.savefig(filename, dpi=self.dpi)
            plt.close()

        piv_test = adanet_test.pivot(
            index='coordinate_from', columns='coordinate_to', values='pearson_avg')
        piv_train = adanet_train.pivot(
            index='coordinate_from', columns='coordinate_to', values='pearson_avg')
        piv = piv_train - piv_test

        overfit = piv.stack().reset_index()
        overfit = overfit.rename(index=str, columns={0: 'overfit_pearson_avg'})
        odf = pd.merge(adanet_test, overfit, how='left',
                       left_on=['coordinate_from', 'coordinate_to'],
                       right_on=['coordinate_from', 'coordinate_to'])
        odf = odf[(odf.coordinate_from != odf.coordinate_to)]
        odf['pair'] = odf['coordinate_from'].apply(
            lambda x: x[:2]) + "_" + odf['coordinate_to'].apply(lambda x: x[:2])
        # odf['capped_train_size'] = np.minimum(odf.train_size,20000)
        odf['log10_train_size'] = np.log(odf.train_size)
        odf['pearson_avg_train'] = odf.overfit_pearson_avg + odf.pearson_avg

        # sns.set_style("whitegrid")
        sns.set_context("talk")
        order = sorted(odf.coordinate_to.unique())
        sns.relplot(x="pearson_avg_train", y="pearson_avg",
                    hue='coordinate_from', hue_order=order,
                    palette=self.cc_palette(order),
                    col='coordinate_to', col_wrap=5, col_order=order,
                    size="log10_train_size", sizes=(5, 100), data=odf,
                    facet_kws={'xlim': (0., 1.), 'ylim': (0., 1.)})
        filename = os.path.join(
            self.plot_path, "sign3_overfit_vs_trainsize.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    @staticmethod
    def spy_sign2_universe_matrix(self, universe_h5, datasets,
                                  chunk_size=1000):

        # the matrix is too big for any plotting attempt
        # we go by density bins
        bins = list()
        with h5py.File(universe_h5, 'r') as hf:
            for i in tqdm(range(0, hf['x_test'].shape[0], chunk_size)):
                chunk = slice(i, i + chunk_size)
                matrix = hf['x_test'][chunk]
                presence = (~np.isnan(matrix[:, 0::128])).astype(int)
                curr_bin = np.sum(presence, axis=0) / float(presence.shape[0])
                bins.append(curr_bin)
        binned = np.vstack(bins)

        # do some column-wise smoothing
        def smooth_fn(r):
            s = interpolate.interp1d(np.arange(len(r)), r)
            xnew = np.arange(0, len(r) - 1, .1)
            return s(xnew)

        smooth = np.vstack([smooth_fn(c) for c in binned.T]).T

        # plot
        sns.set_context('talk')
        # sns.set_style("white")
        fig, ax = plt.subplots(figsize=(14, 12))
        cmap = matplotlib.cm.viridis
        cmap.set_bad(cmap.colors[0])
        im = ax.imshow(smooth * 100, aspect='auto',
                       norm=matplotlib.colors.LogNorm(),
                       cmap=cmap)
        # ax.yaxis.set_xticklabels(
        #    np.arange(0, binned.shape[0] * chunk_size, chunk_size))
        ax.set_xlabel("Bioativity Spaces")
        ax.xaxis.set_label_position('top')
        ax.set_xticks(np.arange(binned.shape[1]))
        ax.set_xticklabels([ds[:2] for ds in datasets])
        ax.tick_params(labelbottom='off', labeltop='on')
        ax.xaxis.labelpad = 20
        ax.set_yticklabels([])
        thousands = binned.shape[0] * chunk_size / 1000.
        ax.set_ylabel("%ik Molecules" % thousands)

        # colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Coverage')
        cbar.ax.yaxis.set_label_position('left')
        cbar.set_ticks([0.01, 0.1, 1., 10, 100])
        cbar.set_ticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
        '''
        # also mark dataset medians
        ds_avg = np.median(binned, axis=0) * 100
        cbar2 = plt.colorbar(im, ax=ax)
        cbar2.ax.set_ylabel('Coverage')
        cbar2.ax.yaxis.set_label_position('left')
        cbar2.set_ticks(ds_avg[np.argsort(ds_avg)])
        cbar2.set_ticklabels(
            np.array([ds[:2] for ds in datasets])[np.argsort(ds_avg)])
        '''

        # save
        plt.tight_layout()
        filename = os.path.join("sign2_universe.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def spy_augment(self, matrix, augment_fn, epochs=1):
        nr_samples, nr_features = matrix.shape
        fig, axes = plt.subplots(nrows=epochs // 5, ncols=5, figsize=(15, 15))
        for idx in range(epochs):
            ax = axes.flatten()[idx]
            aug_mat, _ = augment_fn(matrix, True)
            ax.spy(aug_mat)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        filename = os.path.join(self.plot_path, "spy.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def sign3_adanet_performance_all_plot(self, metric="pearson", suffix=None,
                                          stat_filename="stats_eval.pkl"):

        # sns.set_style("whitegrid")
        # sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
        fig, axes = plt.subplots(25, 1, sharey=True, sharex=False,
                                 figsize=(20, 70), dpi=self.dpi)
        adanet_dir = 'adanet_eval'
        if suffix is not None:
            adanet_dir = 'adanet_%s' % suffix
        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            perf_file = os.path.join(
                s3.model_path, adanet_dir, stat_filename)
            if not os.path.isfile(perf_file):
                continue
            df = pd.read_pickle(perf_file)
            sns.barplot(x='from', y=metric, data=df, hue="split",
                        hue_order=['train', 'test'],
                        ax=ax, color=self.cc_palette([ds])[0])
            ax.set_ylim(0, 1)
            sns.stripplot(x='from', y='coverage', data=df, hue="split",
                          hue_order=['train', 'test'],
                          ax=ax, jitter=False, palette=['pink', 'crimson'],
                          alpha=.9)
            ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_ylabel(ds)
            for idx, p in enumerate(ax.patches):
                if "%.2f" % p.get_height() == 'nan':
                    continue
                val = "%.2f" % p.get_height()
                ax.annotate(val[1:],
                            (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='center', fontsize=11,
                            color='k', rotation=90, xytext=(0, 20),
                            textcoords='offset points')

            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
        plt.tight_layout()
        filename = os.path.join(self.plot_path, "adanet_performance_all.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

    def sign3_adanet_performance_overall(self, metric="pearson", suffix=None,
                                         not_self=True):

        # sns.set_style("whitegrid")
        sns.set_context("talk")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=False,
                                 figsize=(10, 10), dpi=self.dpi)
        adanet_dir = 'adanet_eval'
        if suffix is not None:
            adanet_dir = 'adanet_%s' % suffix
        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            perf_file = os.path.join(
                s3.model_path, adanet_dir, 'stats.pkl')
            if not os.path.isfile(perf_file):
                continue
            df = pd.read_pickle(perf_file)
            if not_self:
                if ds in ['B4.001', 'C3.001', 'C4.001', 'C5.001']:
                    df = df[df['from'] == 'not-BX|CX']
                else:
                    df = df[df['from'] == 'not-%s' % ds]
            sns.barplot(x='from', y=metric, data=df, hue="split",
                        hue_order=['train', 'test'], alpha=.8,
                        ax=ax, color=self.cc_palette([ds])[0])
            ax.set_ylim(0, 1)
            ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.set_xticklabels([ds])
            for idx, p in enumerate(ax.patches):
                if "%.2f" % p.get_height() == 'nan':
                    continue
                val = "%.2f" % p.get_height()
                ax.annotate(val[1:],
                            (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='center', fontsize=11,
                            color='k', rotation=90, xytext=(0, 20),
                            textcoords='offset points')

            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        if suffix is not None:
            filename = os.path.join(
                self.plot_path, "adanet_performance_%s.png" % suffix)
        else:
            filename = os.path.join(
                self.plot_path, "adanet_performance_overall.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

    def sign3_adanet_performance_overall_heatmap(self, metric="pearson",
                                                 split='test',
                                                 suffix=None, not_self=True):
        adanet_dir = 'adanet_eval'
        if suffix is not None:
            adanet_dir = 'adanet_%s' % suffix
        df = pd.DataFrame()
        for ds in tqdm(self.datasets):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            perf_file = os.path.join(s3.model_path, adanet_dir,
                                     'stats_eval.pkl')
            if not os.path.isfile(perf_file):
                continue
            sdf = pd.read_pickle(perf_file)
            sel = sdf[(sdf['split'] == split)].groupby(
                'from', as_index=False)[metric].mean()
            sel['to'] = ds[:2]
            df = pd.concat([ df, sel], ignore_index=True)

        df['from'] = df['from'].map({ds: ds[:2] for ds in self.cc.datasets})
        df = df.dropna()

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=self.dpi)
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(df.pivot('from', 'to', metric), vmin=0, vmax=1,
                    linewidths=.5, square=True, cmap=cmap)
        plt.title('set: %s, metric: %s' % (split, metric))
        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "adanet_perf_heatmap_%s_%s.png" % (split, metric))
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

    def sign3_coverage_heatmap(self, sign2_coverage):
        cov = sign2_coverage.get_h5_dataset('V')
        df = pd.DataFrame(columns=['from', 'to', 'coverage'])
        for ds_from, ds_to in tqdm(itertools.product(self.datasets, self.datasets)):
            idx_from = self.datasets.index(ds_from)
            idx_to = self.datasets.index(ds_to)
            mask_to = cov[:, idx_to].astype(bool)
            tot_to = np.count_nonzero(mask_to)
            having_from = np.count_nonzero(cov[mask_to, idx_from])
            coverage = having_from / float(tot_to)
            df.loc[len(df)] = pd.Series({
                'from': ds_from[:2],
                'to': ds_to[:2],
                'coverage': coverage})

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=self.dpi)
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(df.pivot('from', 'to', 'coverage'), vmin=0, vmax=1,
                    linewidths=.5, square=True, cmap=cmap)
        plt.title('Coverage')
        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "sign3_coverage_heatmap.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

    def sign3_coverage_barplot(self, sign2_coverage):
        # sns.set_style(*self.style)
        cov = sign2_coverage.get_h5_dataset('V')
        df = pd.DataFrame(columns=['from', 'to', 'coverage'])
        for ds_from, ds_to in tqdm(itertools.product(self.datasets, self.datasets)):
            idx_from = self.datasets.index(ds_from)
            idx_to = self.datasets.index(ds_to)
            mask_to = cov[:, idx_to].astype(bool)
            tot_to = np.count_nonzero(mask_to)
            having_from = np.count_nonzero(cov[mask_to, idx_from])
            coverage = having_from / float(tot_to)
            df.loc[len(df)] = pd.Series({
                'from': ds_from[:2],
                'to': ds_to[:2],
                'coverage': coverage})

        cross_cov = df.pivot('from', 'to', 'coverage').values
        fracs = (cross_cov / np.sum(cross_cov, axis=0)).T
        totals = dict()
        for ds in self.datasets:
            idx = self.datasets.index(ds)
            coverage_col = cov[:, idx].astype(bool)
            totals[ds] = np.count_nonzero(coverage_col)

        columns = ['dataset'] + self.datasets
        df2 = pd.DataFrame(columns=columns)
        for ds, frac in list(zip(self.datasets, fracs))[::-1]:
            row = zip(columns, [ds[:2]] +
                      (frac * np.log10(totals[ds])).tolist())
            df2.loc[len(df2)] = pd.Series(dict(row))
        df2.set_index('dataset', inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=(3, 10), dpi=self.dpi)
        colors = [
            '#EA5A49', '#EE7B6D', '#EA5A49', '#EE7B6D', '#EA5A49',
            '#C189B9', '#B16BA8', '#C189B9', '#B16BA8', '#C189B9',
            '#5A72B5', '#7B8EC4', '#5A72B5', '#7B8EC4', '#5A72B5',
            '#96BF55', '#7CAF2A', '#96BF55', '#7CAF2A', '#96BF55',
            '#F39426', '#F5A951', '#F39426', '#F5A951', '#F39426']
        df2.plot.barh(stacked=True, ax=ax, legend=False, color=colors, lw=0)
        sns.despine(left=True, trim=True)
        plt.tick_params(left=False)
        ax.set_ylabel('')
        ax.set_xlabel('Molecule Coverage')
        # ax.xaxis.tick_top()
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xticklabels(
            [r'$10^{0}$', r'$10^{2}$', r'$10^{4}$', r'$10^{6}$', ])
        ax.tick_params(labelsize=14)
        ax.grid(False)

        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "sign3_coverage_barplot.png")
        plt.savefig(filename, dpi=self.dpi)
        if self.svg:
            plt.savefig(filename.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

        fig = plt.figure(figsize=(3, 10))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02,
                            top=0.95, hspace=0.1)
        gs = fig.add_gridspec(2, 1)
        gs.set_height_ratios((50, 1))
        gs_ds = gs[0].subgridspec(1, 2, wspace=0.1, hspace=0.0)
        ax_cov = fig.add_subplot(gs_ds[0])
        ax_xcov = fig.add_subplot(gs_ds[1])

        cdf = pd.DataFrame([(x[:2], totals[x]) for x in sorted(totals.keys(), reverse=True)],
                           columns=['dataset', 'coverage'])
        ax_cov.barh(range(25), cdf['coverage'],
                    color=list(reversed(colors)), lw=0)
        for y, name in enumerate(cdf['dataset'].tolist()):
            ax_cov.text(2, y - 0.06, name, size=12,
                        va='center', ha='left',
                        color='white', fontweight='bold')
        ax_cov.set_yticks(range(1, 26))
        ax_cov.set_yticklabels([])
        # ax_cov.set_xlim(0, 110)
        # ax_cov.xaxis.set_label_position('top')
        # ax_cov.xaxis.tick_top()
        plt.tick_params(left=False)
        ax_cov.set_ylabel('')
        ax_cov.set_xlabel('Molecules', fontsize=16)
        ax_cov.set_xscale('log')
        ax_cov.set_xlim(1, 1e6)
        ax_cov.set_ylim(-1, 25)
        # ax.xaxis.tick_top()
        ax_cov.set_xticks([1e1, 1e3, 1e5])
        # ax_cov.set_xticklabels(
        #    [r'$\mathregular{10^{1}}$', r'$\mathregular{10^{3}}$',
        #     r'$\mathregular{10^{5}}$', ])
        ax_cov.grid(False)
        # ax_cov.tick_params(labelsize=14)
        ax_cov.tick_params(left=False, labelsize=14, pad=0, direction='inout')
        sns.despine(ax=ax_cov, left=True, bottom=True, top=False, trim=True)
        ax_cov.xaxis.set_label_position('top')
        ax_cov.xaxis.tick_top()
        ax_xcov.tick_params(left=False, bottom=False, top=True,
                            labelbottom=False, labeltop=True)
        cmap_tmp = plt.get_cmap("magma", 14)
        cmap = matplotlib.colors.ListedColormap(
            cmap_tmp(np.linspace(0, 1, 14))[4:], 10)
        for i in range(25):
            ax_xcov.barh(25 - i, 1, left=range(25),
                         color=[cmap(1 - cross_cov[i, x]) for x in range(25)],
                         lw=0)
        ax_xcov.set_ylim(0, 26)
        ax_xcov.set_yticks([])
        ax_xcov.set_yticklabels([])
        ax_xcov.set_xticks([])
        ax_xcov.set_xticklabels([])
        ax_xcov.set_yticks(range(1, 26))

        for y, color in enumerate(colors):
            rect = matplotlib.patches.Rectangle(
                (y, 25.75), 1, 0.5, lw=0, edgecolor='w', facecolor=color,
                clip_on=False)
            ax_xcov.add_patch(rect)
        main_colors = ['#EA5A49', '#B16BA8', '#5A72B5', '#7CAF2A', '#F39426']
        for y, col in enumerate(main_colors):
            ax_xcov.set_yticks(range(1, 26))
            rect = matplotlib.patches.Rectangle(
                (y * 5, 26), 5, 1, lw=0.1, edgecolor='w', facecolor=col,
                clip_on=False)
            ax_xcov.add_patch(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0
            ax_xcov.text(cx, cy, 'ABCDE'[y], weight='bold',
                         size=12, ha='center', va='center',
                         color='white', clip_on=False)

        sns.despine(ax=ax_xcov, left=True, bottom=True)
        ax_xcov.tick_params(left=False, bottom=False, top=False,
                            labelbottom=False, labeltop=False)
        ax_xcov.grid(False)
        ax_cbar = fig.add_subplot(gs[1])
        cbar = matplotlib.colorbar.ColorbarBase(
            ax_cbar, cmap=cmap, orientation='horizontal',
            ticklocation='top')
        cbar.ax.set_xlabel('Overlap', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([1, .8, .6, .4, .2, .0])
        cbar.set_ticklabels(['0', '', '', '', '', '1'])

        poly = Polygon([(0.05, 2.0), (0.05, 1.4), (1, 1.4)],
                       transform=ax_cbar.get_xaxis_transform(),
                       clip_on=False,
                       facecolor='black')
        ax_cbar.add_patch(poly)

        outfile = os.path.join(self.plot_path, 'sign3_coverage_barplot.png')
        plt.savefig(outfile, dpi=self.dpi)
        if self.svg:
            plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

    def cctype_CCA(self, cctype1='sign4', cctype2='sign1', limit=10000):

        cca_file = os.path.join(
            self.plot_path, "%s_%s_CCA.pkl" % (cctype1, cctype2))
        if not os.path.isfile(cca_file):
            df = pd.DataFrame(columns=['from', 'to', cctype2, cctype1])
            for i in range(len(self.datasets)):
                ds_from = self.datasets[i]
                s2_from = self.cc.get_signature(
                    cctype2, 'full', ds_from)
                s3_from = self.cc.get_signature(
                    cctype1, 'full', ds_from)[:limit]
                for j in range(i + 1):
                    ds_to = self.datasets[j]
                    if ds_to == ds_from:
                        df.loc[len(df)] = pd.Series({
                            'from': ds_from[:2],
                            'to': ds_to[:2],
                            cctype1: 1.0,
                            cctype2: 1.0})
                        continue
                    s2_to = self.cc.get_signature(
                        cctype2, 'full', ds_to)
                    s3_to = self.cc.get_signature(
                        cctype1, 'full', ds_to)[:limit]
                    s3_res = dataset_correlation(s3_from, s3_to)

                    # shared keys
                    shared_inks = s2_from.unique_keys & s2_to.unique_keys
                    shared_inks = sorted(list(shared_inks))[:limit]
                    mask_from = np.isin(list(s2_from.keys), list(shared_inks))
                    mask_to = np.isin(list(s2_to.keys), list(shared_inks))
                    ss2_from = s2_from[
                        :limit * 10][mask_from[:limit * 10]][:limit]
                    ss2_to = s2_to[:limit * 10][mask_to[:limit * 10]][:limit]
                    min_size = min(len(ss2_to), len(ss2_from))
                    print(ds_from, ds_to, 'S2 min_size', min_size)
                    if min_size < 10:
                        df.loc[len(df)] = pd.Series({
                            'from': ds_from[:2],
                            'to': ds_to[:2],
                            cctype1: s3_res[0],
                            cctype2: 0.0})
                        df.loc[len(df)] = pd.Series({
                            'from': ds_to[:2],
                            'to': ds_from[:2],
                            cctype1: s3_res[3],
                            cctype2: 0.0})
                        continue
                    s2_res = dataset_correlation(
                        ss2_from[:min_size], ss2_to[:min_size])

                    df.loc[len(df)] = pd.Series({
                        'from': ds_from[:2],
                        'to': ds_to[:2],
                        cctype1: s3_res[0],
                        cctype2: s2_res[0]})

                    df.loc[len(df)] = pd.Series({
                        'from': ds_to[:2],
                        'to': ds_from[:2],
                        cctype1: s3_res[3],
                        cctype2: s2_res[3]})
                    print(ds_from, ds_to, 's3 %.2f' %
                          s3_res[0], 's2 %.2f' % s2_res[0])
            df.to_pickle(cca_file)
        df = pd.read_pickle(cca_file)

        # sns.set_style("ticks")
        # sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        # CCA heatmaps
        for cca_id in [cctype2, cctype1]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=self.dpi)
            cmap = plt.cm.get_cmap('plasma_r', 5)
            sns.heatmap(df.pivot('from', 'to', cca_id), vmin=0, vmax=1,
                        linewidths=.2, square=True, cmap=cmap, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.title('Canonical Correlation Analysis')
            plt.tight_layout()
            filename = os.path.join(
                self.plot_path, "%s_%s_heatmap.png" % (cctype1, cca_id))
            plt.savefig(filename, dpi=self.dpi)
            plt.close('all')

        # combined heatmap
        cca3 = df.pivot('from', 'to', cctype1).values
        cca3_avg = np.zeros_like(cca3)
        for i,  j in itertools.product(range(25), range(25)):
            cca3_avg[i, j] = np.mean((cca3[i, j], cca3[j, i]))
        cca2 = df.pivot('from', 'to', cctype2).values
        cca2_avg = np.zeros_like(cca2)
        for i,  j in itertools.product(range(25), range(25)):
            cca2_avg[i, j] = np.mean((cca2[i, j], cca2[j, i]))
        combined = np.tril(cca3_avg) + np.triu(cca2_avg)
        np.fill_diagonal(combined, np.nan)

        fig = plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.9,
                            wspace=.02, hspace=.02)
        gs = fig.add_gridspec(2, 2)
        gs.set_height_ratios((1, 10))
        gs.set_width_ratios((20, 1))
        ax_dist = fig.add_subplot(gs[0, 0])
        cca2_dist = cca2_avg[np.triu_indices_from(cca2_avg, 1)]
        cca3_dist = cca3_avg[np.tril_indices_from(cca3_avg, -1)]
        sns.kdeplot(cca2_dist, ax=ax_dist, shade=True, bw=.2, color='blue',
                    clip=(min(cca2_dist), max(cca2_dist)))
        sns.kdeplot(cca3_dist, ax=ax_dist, shade=True, bw=.2, color='red',
                    clip=(min(cca3_dist), max(cca3_dist)))
        ax_dist.text(0.05, 0.35, "Observed", transform=ax_dist.transAxes,
                     name='Arial', size=14, color='blue')
        ax_dist.text(0.05, 0.8, "Predicted", transform=ax_dist.transAxes,
                     name='Arial', size=14, color='red')
        fig.text(0.5, 1.7, 'Canonical Correlation Analysis',
                 ha='center', va='center', transform=ax_dist.transAxes,
                 name='Arial', size=16)
        # sns.despine(ax=ax_dist, left=True, bottom=True)
        ax_dist.set_axis_off()
        ax_hm = fig.add_subplot(gs[1, 0])
        ax_cbar = fig.add_subplot(gs[1, 1])
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(combined, vmin=0, vmax=1,
                    linewidths=.2, square=True, cmap=cmap,
                    ax=ax_hm, cbar_ax=ax_cbar)
        ax_hm.set_xlabel('')
        ax_hm.set_ylabel('')
        ax_hm.set_xticklabels([x[:2] for x in self.datasets], rotation=90)
        ax_hm.set_yticklabels([x[:2] for x in self.datasets], rotation=0)
        bottom, top = ax_hm.get_ylim()
        ax_hm.set_ylim(bottom + 0.5, top - 0.5)
        filename = os.path.join(
            self.plot_path, "%s_%s_CCA_heatmap.png" % (cctype1, cctype2))
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

        # scatter plot comparing sign1 and sign4
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=self.dpi)
        sns.scatterplot(x=cctype2, y=cctype1, data=df, ax=ax)
        filename = os.path.join(
            self.plot_path, "%s_%s_CCA_scatter.png" % (cctype1, cctype2))
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=self.dpi)
        for i in range(len(self.datasets)):
            for j in range(len(self.datasets)):
                dsx = self.datasets[i][:2]
                dsy = self.datasets[j][:2]
                if dsx == dsy:
                    continue
                c1 = self.cc_colors(dsx)
                c2 = self.cc_colors(dsy)
                marker_style = dict(color=c1,  markerfacecoloralt=c2,
                                    markeredgecolor='w', markeredgewidth=0,
                                    markersize=8, marker='o')
                x = df[(df['from'] == dsx) & (df['to'] == dsy)][
                    cctype2].tolist()[0]
                y = df[(df['from'] == dsx) & (df['to'] == dsy)][
                    cctype1].tolist()[0]
                ax.plot(x, y, alpha=0.9, fillstyle='right', **marker_style)
        ax.set_xlabel('%s CCA' % cctype2,
                      fontdict=dict(name='Arial', size=16))
        ax.set_ylabel('%s CCA' % cctype1,
                      fontdict=dict(name='Arial', size=16))
        ax.tick_params(labelsize=14)
        sns.despine(ax=ax, trim=True)
        filename = os.path.join(
            self.plot_path, "%s_%s_CCA_comparison.png" % (cctype1, cctype2))
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

        # also plot as MDS projection
        cca = df.pivot('from', 'to', cctype1).values
        # make average of upper and lower triangular matrix
        cca_avg = np.zeros_like(cca)
        for i,  j in itertools.product(range(25), range(25)):
            cca_avg[i, j] = np.mean((cca[i, j], cca[j, i]))
        proj = MDS(dissimilarity='precomputed', random_state=0)
        coords = proj.fit_transform(1 - cca_avg)

        def anno_dist(ax, idx1, idx2, coords):
            p1 = coords[idx1]
            p2 = coords[idx2]
            coords_dist = dist = np.linalg.norm(p1 - p2)
            dist = 1 - cca_avg[idx1, idx2]
            angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
            if angle < 0:
                angle += 5
            else:
                angle -= 5
            ax.annotate(
                '', xy=p1, xycoords='data',
                xytext=p2, textcoords='data',
                zorder=1,
                arrowprops=dict(
                    arrowstyle="-", color="0.2",
                    shrinkA=10, shrinkB=10,
                    patchA=None, patchB=None,
                    connectionstyle="bar,angle=%.2f,fraction=-%.2f" % (
                        angle, (1 - np.power(coords_dist, 1 / 3))),
                ),)

            # ax.plot([p1[0],p2[0]], [p1[1],p2[1]])
            midpoint = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.annotate('%.2f' % dist, xy=midpoint,
                        xytext=midpoint, textcoords='data', xycoords='data',
                        rotation=angle, va='center', ha='center')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=self.dpi)
        colors = [self.cc_colors(ds) for ds in self.datasets]
        markers = ["$\u2776$", "$\u2777$", "$\u2778$", "$\u2779$", "$\u277a$"]
        for i, color in enumerate(colors):
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=400,
                       edgecolor='', marker=markers[i % 5], zorder=2)
        anno_dist(ax, 19, 15, coords)
        # anno_dist(ax, 20, 22,coords)
        anno_dist(ax, 1, 2, coords)
        anno_dist(ax, 9, 20, coords)
        filename = os.path.join(
            self.plot_path, "%s_CCA_MDS.svg" % cctype1)
        plt.axis('off')
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')

    def all_sign_validations(self, sign_types=None, molsets=None, valset='moa'):

        if sign_types is None:
            sign_types = ['sign1', 'sign2', 'sign3']
        if molsets is None:
            molsets = ['reference', 'full']
            #['atc_auc', 'atc_cov', 'atc_ks_d', 'atc_ks_p',
            #           'moa_auc', 'moa_cov', 'moa_ks_d', 'moa_ks_p']
        df = pd.DataFrame(
            columns=['sign_molset', 'dataset', 'metric', 'value'])
        for ds in self.datasets:
            for molset in molsets:
                for sign_type in sign_types:
                    try:
                        sign = self.cc.get_signature(sign_type, molset, ds)
                    except Exception as err:
                        self.__log.warning(
                            "Skippin %s: %s", str(sign), str(err))
                        continue
                    stat_file = os.path.join(
                        sign.stats_path, 'validation_stats.json')
                    if not os.path.isfile(stat_file):
                        continue
                    stats = json.load(open(stat_file, 'r'))
                    for k, v in stats.items():
                        row = {
                            'sign_molset': '_'.join([sign_type, molset]),
                            'dataset': ds,
                            'metric': k,
                            'value': float(v),
                        }
                        if 'cov' in k:
                            row['value'] /= 100.
                        df.loc[len(df)] = pd.Series(row)
        print(df)
        # sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(15, 15), dpi=self.dpi)
        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):
            ds_color = self.cc_palette([ds])[0]
            sns.barplot(x='sign_molset', y='value',
                        data=df[(df.dataset == ds) & (
                            df.metric == '%s_auc' % valset)],
                        ax=ax, alpha=1,
                        color=ds_color)

            sns.stripplot(x='sign_molset', y='value',
                          data=df[(df.dataset == ds) & (
                              df.metric == '%s_cov' % valset)],
                          size=10, marker="o", edgecolor='k', linewidth=2,
                          ax=ax, jitter=False, alpha=1, color='w')
            ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.set_xticklabels([ds])
            for idx, p in enumerate(ax.patches):
                if "%.2f" % p.get_height() == 'nan':
                    continue
                val = p.get_height()
                if val > 1.0:
                    val = "%.1f" % p.get_height()
                else:
                    val = "%.2f" % p.get_height()
                ax.annotate(val,
                            (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='center', fontsize=11,
                            color='k', rotation=90, xytext=(0, 20),
                            textcoords='offset points')
            if ds.startswith('E'):
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                    label.set_rotation(45)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle="-",
                    color=ds_color, lw=0.3)
            ax.spines["bottom"].set_color(ds_color)
            ax.spines["top"].set_color(ds_color)
            ax.spines["right"].set_color(ds_color)
            ax.spines["left"].set_color(ds_color)
        plt.tight_layout()
        filename = os.path.join(self.plot_path,
                                "sign_validation_%s_%s_%s.png"
                                % (valset, '_'.join(sign_types),
                                    '_'.join(molsets)))
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')
        """
        for metric in metrics:
            # sns.set_style("whitegrid")
            fig, axes = plt.subplots(5, 5, sharey=True, sharex=False,
                                     figsize=(10, 10), dpi=self.dpi)
            for ds, ax in tqdm(zip(self.cc.datasets, axes.flatten())):

                cdf = df[df.dataset == ds][metric]
                s1v = cdf[df.sign_type == 'sign1'].iloc[0]
                s2v = cdf[df.sign_type == 'sign2'].iloc[0]
                s3v = cdf[df.sign_type == 'sign3'].iloc[0]

                sns.barplot(x=['s2-s1', 's3-s1'], y=[100 * (s2v - s1v) / s1v, 100 * (s3v - s1v) / s1v],
                            ax=ax, alpha=.8, color=self.cc_palette([ds])[0])
                ax.set_xlabel('')
                ax.set_ylabel('')
                # ax.set_xticklabels([ds])
                for idx, p in enumerate(ax.patches):
                    if "%.2f" % p.get_height() == 'nan':
                        continue
                    val = p.get_height()
                    if val > 1.0:
                        val = "%.1f" % p.get_height()
                    else:
                        val = "%.2f" % p.get_height()
                    ax.annotate(val,
                                (p.get_x() + p.get_width() / 2., 0),
                                ha='center', va='center', fontsize=11,
                                color='k', rotation=90, xytext=(0, 20),
                                textcoords='offset points')

                ax.grid(axis='y', linestyle="-",
                        color=self.cc_palette([ds])[0], lw=0.3)
                ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
                ax.spines["top"].set_color(self.cc_palette([ds])[0])
                ax.spines["right"].set_color(self.cc_palette([ds])[0])
                ax.spines["left"].set_color(self.cc_palette([ds])[0])
            plt.tight_layout()
            filename = os.path.join(self.plot_path,
                                    "sign_validation_deltas_%s.png" % metric)
            plt.savefig(filename, dpi=self.dpi)
            plt.close('all')
        """

    def cctype_similarity_search(self, cctype='sign4', cctype_ref='sign1',
                                 limit=10000, limit_neig=50000, sign_cap=200000):

        from chemicalchecker.core.signature_data import DataSignature
        import faiss

        def mask_exclude(idxs, x1_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            return x1_data_transf

        def mask_keep(idxs, x1_data):
            # we will fill an array of NaN with values we want to keep
            x1_data_transf = np.zeros_like(x1_data, dtype=float) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = x1_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            return x1_data_transf

        def background_distances(matrix, metric, sample_pairs=100000, unflat=True,
                                 memory_safe=False):

            PVALRANGES = np.array([0, 0.001, 0.01, 0.1] +
                                  list(np.arange(1, 100)) + [100]) / 100.
            metric_fn = eval(metric)

            if matrix.shape[0]**2 < sample_pairs:
                print("Requested more pairs then possible combinations")
                sample_pairs = matrix.shape[0]**2 - matrix.shape[0]

            bg = list()
            done = set()
            tries = 1e6
            tr = 0
            while len(bg) < sample_pairs and tr < tries:
                tr += 1
                i = np.random.randint(0, matrix.shape[0] - 1)
                j = np.random.randint(i + 1, matrix.shape[0])
                if (i, j) not in done:
                    dist = metric_fn(matrix[i], matrix[j])
                    bg.append(dist)
                    done.add((i, j))
            # pavalues as percentiles
            i = 0
            PVALS = [(0, 0., i)]  # DISTANCE, RANK, INTEGER
            i += 1
            percs = PVALRANGES[1:-1] * 100
            for perc in percs:
                PVALS += [(np.percentile(bg, perc), perc / 100., i)]
                i += 1
            PVALS += [(np.max(bg), 1., i)]
            # prepare returned dictionary
            bg_distances = dict()
            if not unflat:
                bg_distances["distance"] = np.array([p[0] for p in PVALS])
                bg_distances["pvalue"] = np.array([p[1] for p in PVALS])
            else:
                # Remove flat regions whenever we observe them
                dists = [p[0] for p in PVALS]
                pvals = np.array([p[1] for p in PVALS])
                top_pval = np.min(
                    [1. / sample_pairs, np.min(pvals[pvals > 0]) / 10.])
                pvals[pvals == 0] = top_pval
                pvals = np.log10(pvals)
                dists_ = sorted(set(dists))
                pvals_ = [pvals[dists.index(d)] for d in dists_]
                dists = np.interp(pvals, pvals_, dists_)
                thrs = [(dists[t], PVALS[t][1], PVALS[t][2])
                        for t in range(len(PVALS))]
                bg_distances["distance"] = np.array([p[0] for p in thrs])
                bg_distances["pvalue"] = np.array([p[1] for p in thrs])
            return bg_distances

        def jaccard_similarity(n1, n2):
            """Compute Jaccard similarity."""
            s1 = set(n1)
            s2 = set(n2)
            inter = len(set.intersection(s1, s2))
            uni = len(set.union(s1, s2))
            return inter / float(uni)

        def overlap(n1, n2):
            """Compute Overlap."""
            s1 = set(n1)
            s2 = set(n2)
            uni = len(set.intersection(s1, s2))
            return float(uni) / len(s1)

        outfile = os.path.join(
            self.plot_path, '%s_simsearch.pkl' % cctype)
        if not os.path.isfile(outfile):
            df = pd.DataFrame(
                columns=['dataset', 'nthr', 'cthr', 'dthr',
                         'log-odds-ratio', 'logodds_err'])
            # [(5,'B1.001'),(15,'D1.001')]:
            for ds_idx, ds in list(enumerate(self.datasets)):
                sign = self.cc.get_signature(cctype, 'full', ds)
                signref = self.cc.get_signature(cctype_ref, 'full', ds)
                # get siamese train/test inks
                traintest_file = os.path.join(
                    sign.model_path, 'traintest_eval.h5')
                tt = DataSignature(traintest_file)
                train_inks = tt.get_h5_dataset('keys_train')[:limit_neig]
                train_inks = np.sort(train_inks)
                train_mask = np.isin(
                    list(signref.keys), list(train_inks), assume_unique=True)
                test_inks = tt.get_h5_dataset('keys_test')[:limit]
                test_inks = np.sort(test_inks)
                test_mask = np.isin(
                    list(signref.keys), list(test_inks), assume_unique=True)
                # get train/test sign1
                slice_cap = slice(0, sign_cap)
                signref_V = signref.get_h5_dataset('V', mask=slice_cap)
                train_signref = signref_V[train_mask[slice_cap]]
                test_signref = signref_V[test_mask[slice_cap]]
                # predict train/test sign4
                print('REFERENCE SIGNATURE:', cctype_ref)
                print('train_signref', train_signref.shape)
                print('test_signref', test_signref.shape)
                # predict train/test sign4
                input_file = DataSignature(
                    os.path.join(sign.model_path, 'train.h5'))
                input_x = input_file.get_h5_dataset('x', mask=slice_cap)
                train_input = input_x[train_mask[slice_cap]]
                test_input = input_x[test_mask[slice_cap]]
                print('PREDICTION INPUT:', traintest_file)
                print('train_input', train_input.shape)
                print('test_input', test_input.shape)
                # laod eval siamese predictor
                predict_fn = sign.get_predict_fn(
                    smiles=False, model='siamese_eval')
                train_sign = predict_fn(train_input)
                test_sign = predict_fn(mask_exclude([ds_idx], test_input))
                print('PREDICTION OUTPUT:', traintest_file)
                print('train_sign', train_sign.shape)
                print('test_sign', test_sign.shape)
                # get confidence for test sign4
                conf_mask = np.isin(
                    list(sign.keys), list(test_inks), assume_unique=True)
                test_confidence = sign.get_h5_dataset(
                    'confidence')[conf_mask][:len(test_sign)]
                print('test_confidence', test_confidence.shape)
                # make train sign1 neig
                train_signref_neig = faiss.IndexFlatL2(train_signref.shape[1])
                train_signref_neig.add( np.array(train_signref, dtype='float32') )
                # make train sign4 neig
                train_sign_neig = faiss.IndexFlatL2(train_sign.shape[1])
                train_sign_neig.add( np.array(train_sign, dtype='float32') )
                # find test sign1 neighbors
                signref_neig_dist, signref_neig_idx = train_signref_neig.search(
                    np.array( test_signref, dtype='float32'), 100)
                signref_neig_dist = np.sqrt(signref_neig_dist)
                # find test sign4 neighbors
                sign_neig_dist, sign_neig_idx = train_sign_neig.search(
                    np.array( test_sign, dtype='float32'), 100)
                sign_neig_dist = np.sqrt(sign_neig_dist)
                # check various thresholds
                # get sign ref background distances thresholds
                back = background_distances(train_signref, 'euclidean')
                dthrs = list()
                dthrs.append((back['distance'][1], back['pvalue'][1]))
                dthrs.append((back['distance'][5], back['pvalue'][5]))
                dthrs.append((back['distance'][-1], back['pvalue'][-1]))
                nthrs = [10]  # top neighbors
                cthrs = [0, .5, .8]  # confidence
                for nthr, cthr, dthr in itertools.product(nthrs, cthrs, dthrs):
                    hits = 0
                    rnd_hits = collections.defaultdict(int)
                    for row in tqdm(range(signref_neig_dist.shape[0])):
                        # limit original space neighbors by distance
                        d_mask = signref_neig_dist < dthr[0]
                        # limit signature molecule by confidence
                        c_mask = test_confidence > cthr
                        if not c_mask[row]:
                            continue
                        # select top n valid neighbors
                        ref_neig = signref_neig_idx[row][d_mask[row]][:nthr]
                        # if no neighbors we skip the molecule
                        if len(ref_neig) == 0:
                            continue
                        # compare to sign neighbors
                        sign_neig = sign_neig_idx[row][:nthr]
                        hits += len(set(ref_neig).intersection(sign_neig))
                        # compute random background
                        rnd_idxs = np.arange(train_signref_neig.ntotal)
                        for i in range(1000):
                            rnd_neig = np.random.choice(rnd_idxs, nthr)
                            rnd_hits[
                                i] += len(set(ref_neig).intersection(rnd_neig))

                    rnd_hits = [v for k, v in rnd_hits.items()]
                    rnd_mean = np.mean(rnd_hits)
                    rnd_std = np.std(rnd_hits)
                    logodds = np.log2(hits / rnd_mean)
                    logodds_std = np.log2(hits / (rnd_mean + rnd_std))
                    logodds_err = abs(logodds - logodds_std)
                    print(nthr, cthr, dthr, 'log-odds-ratio', logodds)
                    df.loc[len(df)] = pd.Series({
                        'dataset': ds,
                        'nthr': nthr,
                        'cthr': cthr,
                        'dthr': dthr[1],
                        'log-odds-ratio': logodds,
                        'logodds_err': logodds_err,
                    })
            df.to_pickle(outfile)
        df = pd.read_pickle(outfile)

        nthrs = df['nthr'].unique()
        dthrs = df['dthr'].unique()
        max_odds = df['log-odds-ratio'].describe()['75%'] * 1.5
        min_odds = 0
        for nthr, dthr in itertools.product(nthrs, dthrs):
            fdf = df[(df.nthr == nthr) & (df.dthr == dthr)]
            if len(fdf) == 0:
                continue
            fig = plt.figure(constrained_layout=True, figsize=(5, 10))
            gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
            plt.subplots_adjust(left=0.14, right=.92, bottom=0.06, top=.95)
            axes = list()
            for row, col in itertools.product(range(5), range(5)):
                axes.append(fig.add_subplot(gs[row, col]))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False,
                            bottom=False, left=False, right=False)
            plt.grid(False)
            plt.xlabel("Confidence", size=18)
            plt.ylabel("Log-Odds Ratio", size=18)

            for ds, ax in zip(self.datasets[:], axes):
                dsdf = fdf[fdf.dataset == ds]
                if len(dsdf) == 0:
                    continue
                ax.errorbar([1, 2, 3], dsdf['log-odds-ratio'],
                            yerr=dsdf['logodds_err'], fmt='-',
                            color=self.cc_colors(ds, 0),
                            ecolor=self.cc_colors(ds, 1),
                            elinewidth=2, capsize=0)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_ylim(min_odds, max_odds)
                ax.set_xlim(0.5, 3.5)
                x = [str(i) for i in dsdf['cthr'].unique()]
                ax.xaxis.set_ticklabels(x)
                # axis ticks
                if ds[:2] == 'E1':
                    # set the alignment for outer ticklabels
                    ticklabels = ax.get_yticklabels()
                    ticklabels[0].set_va("bottom")
                    ticklabels[-1].set_va("top")
                elif ds[1] == '1':
                    # set the alignment for outer ticklabels
                    ticklabels = ax.get_yticklabels()
                    ticklabels[0].set_va("bottom")
                    ticklabels[-1].set_va("top")
                    ax.xaxis.set_ticklabels([])
                elif ds[0] == 'E':
                    ax.yaxis.set_ticklabels([])
                else:
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])

                # axis labels
                if ds[0] == 'A':
                    ax.set_xlabel(ds[1], fontsize=18, labelpad=15)
                    ax.xaxis.set_label_position('top')
                if ds[1] == '5':
                    ax.set_ylabel(ds[0], fontsize=18, rotation=0, va='center',
                                  labelpad=15)
                    ax.yaxis.set_label_position('right')

            outfile = os.path.join(
                self.plot_path, 'simsearch_%s_%s.png' % (nthr, dthr))
            print(outfile)
            plt.savefig(outfile, dpi=self.dpi)
            plt.close('all')

    def diagnosis_projections(self, cctype):
        fig = plt.figure(constrained_layout=True, figsize=(8, 8))
        gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(left=0.08, right=.95, bottom=0.08, top=.95)
        axes = list()
        for row, col in itertools.product(range(5), range(5)):
            axes.append(fig.add_subplot(gs[row, col]))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Dim 1", size=18)
        plt.ylabel("Dim 2", size=18)

        for ds, ax in zip(self.datasets[:], axes):
            sign = self.cc.get_signature(cctype, 'full', ds)
            diag = DiagnosisPlot(self.cc, sign)
            diag.projection(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title('')
            # axis ticks
            if ds[:2] == 'E1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
                ax.xaxis.set_ticklabels([])
            elif ds[0] == 'E':
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # axis labels
            if ds[0] == 'A':
                ax.set_xlabel(ds[1], fontsize=18, labelpad=15)
                ax.xaxis.set_label_position('top')
            if ds[1] == '5':
                ax.set_ylabel(ds[0], fontsize=18, rotation=0, va='center',
                              labelpad=15)
                ax.yaxis.set_label_position('right')

        outfile = os.path.join(
            self.plot_path, 'diagnosis_projections.png')
        print(outfile)
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def diagnosis_confidences_projection(self, cctype):
        fig = plt.figure(constrained_layout=True, figsize=(7, 7))
        gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(left=0.08, right=.95, bottom=0.08, top=.95)
        axes = list()
        for row, col in itertools.product(range(5), range(5)):
            axes.append(fig.add_subplot(gs[row, col]))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Dim 1", size=18)
        plt.ylabel("Dim 2", size=18)

        for ds, ax in zip(self.datasets[:], axes):
            sign = self.cc.get_signature(cctype, 'full', ds)
            diag = DiagnosisPlot(self.cc, sign)
            diag.confidences_projection(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title('')
            # axis ticks
            if ds[:2] == 'E1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
                ax.xaxis.set_ticklabels([])
            elif ds[0] == 'E':
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # axis labels
            if ds[0] == 'A':
                ax.set_xlabel(ds[1], fontsize=18, labelpad=15)
                ax.xaxis.set_label_position('top')
            if ds[1] == '5':
                ax.set_ylabel(ds[0], fontsize=18, rotation=0, va='center',
                              labelpad=15)
                ax.yaxis.set_label_position('right')

        outfile = os.path.join(
            self.plot_path, 'diagnosis_confidences_projection.png')
        print(outfile)
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def diagnosis_euclidean_distances(self, cctype):
        fig = plt.figure(constrained_layout=True, figsize=(10, 10))
        gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(left=0.08, right=.95, bottom=0.08, top=.95)
        axes = list()
        for row, col in itertools.product(range(5), range(5)):
            axes.append(fig.add_subplot(gs[row, col]))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Euclidean Distance", size=18)
        plt.ylabel("Density", size=18)

        for ds, ax in zip(self.datasets[:], axes):
            sign = self.cc.get_signature(cctype, 'full', ds)
            diag = DiagnosisPlot(self.cc, sign)
            diag.euclidean_distances(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title('')
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 5)
            # axis ticks
            if ds[:2] == 'E1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
                ax.xaxis.set_ticklabels([])
            elif ds[0] == 'E':
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # axis labels
            if ds[0] == 'A':
                ax.set_xlabel(ds[1], fontsize=18, labelpad=15)
                ax.xaxis.set_label_position('top')
            if ds[1] == '5':
                ax.set_ylabel(ds[0], fontsize=18, rotation=0, va='center',
                              labelpad=15)
                ax.yaxis.set_label_position('right')

        outfile = os.path.join(
            self.plot_path, 'diagnosis_moa.png')
        print(outfile)
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign3_neig2_jaccard(self, limit=2000):

        df = pd.DataFrame(columns=['dataset', 'confidence', 'jaccard'])
        for ds in self.datasets[:]:
            s2 = self.cc.get_signature('sign2', 'full', ds)
            n2 = self.cc.get_signature('neig2', 'reference', ds)
            s3 = self.cc.get_signature('sign3', 'full', ds)

            # decide sample molecules
            inks = s2.keys
            s3_conf = s3.get_h5_dataset('confidence')
            s3_intensity = s3.get_h5_dataset('intensity_norm')
            s2_mask = np.isin(list(s3.keys), list(s2.keys), assume_unique=True)
            s2_conf = s3_conf[s2_mask]
            s2_intensity = s3_intensity[s2_mask]

            high_conf = s2_conf > .9
            inks_high = inks[high_conf][:limit]
            # get sign2 and sign3
            _, s2_data = s2.get_vectors(inks_high)
            _, s3_data = s3.get_vectors(inks_high)
            _, s3_data_conf = s3.get_vectors(
                inks_high, dataset_name='confidence')
            s3_data_conf = s3_data_conf.flatten()
            # get idxs of nearest neighbors of s2 and s3
            k = 10
            n2_s2 = n2.get_kth_nearest(
                list(s2_data), k=k, distances=False, keys=False)
            n2_s3 = n2.get_kth_nearest(
                list(s3_data), k=k, distances=False, keys=False)

            jacc = n2.jaccard_similarity(n2_s2['indices'], n2_s3['indices'])
            df = pd.concat([ df, pd.DataFrame(
                {'dataset': ds, 'confidence': 'high', 'jaccard': jacc}) ], ignore_index=True)
            print('***** HIGH', len(jacc), np.mean(jacc),
                  stats.spearmanr(jacc, s3_data_conf))

            all_conf = np.ones_like(s2_conf).astype(bool)
            if len(jacc) < limit:
                new_limit = len(jacc)
            else:
                new_limit = limit
            inks_all = inks[all_conf][:new_limit]
            # get sign2 and sign3
            _, s2_data = s2.get_vectors(inks_all)
            _, s3_data = s3.get_vectors(inks_all)
            _, s3_data_conf = s3.get_vectors(
                inks_all, dataset_name='confidence')
            s3_data_conf = s3_data_conf.flatten()
            # get idxs of nearest neighbors of s2 and s3
            k = 10
            n2_s2 = n2.get_kth_nearest(
                list(s2_data), k=k, distances=False, keys=False)
            n2_s3 = n2.get_kth_nearest(
                list(s3_data), k=k, distances=False, keys=False)

            jacc = n2.jaccard_similarity(n2_s2['indices'], n2_s3['indices'])
            df = pd.concat([ df, pd.DataFrame(
                {'dataset': ds, 'confidence': 'all', 'jaccard': jacc}) ], ignore_index=True)
            print('***** ALL', len(jacc), np.mean(jacc),
                  stats.spearmanr(jacc, s3_data_conf))

        # sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(4, 6), sharex=True, sharey='row')

        plt.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.99,
                            wspace=.08, hspace=.1)
        for ds, ax in zip(self.datasets[:], axes.flat):
            sns.barplot(data=df[df.dataset == ds], y='jaccard', x='confidence',
                        order=['all', 'high'],
                        ax=ax,
                        palette=[self.cc_colors(ds, 2), self.cc_colors(ds, 0)])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(0, 1)
            if ds[:2] == 'E1':
                sns.despine(ax=ax, offset=3, trim=True)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['All', 'High'])
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)
        f.text(0.5, 0.04, 'Confidence', ha='center', va='center')
        f.text(0.06, 0.5, 'Jaccard Similarity', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_neig2_jaccard.png')
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign_property_distribution(self, cctype, molset, prop, xlim=None,
                                   ylim=None, known_delta=True):
        # sns.set_style("whitegrid")
        # sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        fig = plt.figure(constrained_layout=True, figsize=(6, 6))
        gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(left=0.1, right=.95, bottom=0.1, top=.95)
        axes = list()
        for row, col in itertools.product(range(5), range(5)):
            axes.append(fig.add_subplot(gs[row, col]))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel(prop.capitalize(), size=18)
        plt.ylabel("Molecules", size=18)

        for ds, ax in zip(self.datasets, axes):
            try:
                sign = self.cc.get_signature(cctype, molset, ds)
            except Exception:
                continue
            if not os.path.isfile(sign.data_path):
                continue
            # decide sample molecules
            prop_data = sign.get_h5_dataset(prop)
            if known_delta:
                known_mask = sign.get_h5_dataset('known')
                plot_data = [prop_data[known_mask], prop_data[~known_mask]]
                colors = [self.cc_colors(ds, 0), self.cc_colors(ds, 2)]
                print(ds, prop, np.min(plot_data[0]), np.max(plot_data[0]))
                print(ds, prop, np.min(plot_data[1]), np.max(plot_data[1]))
                ax.hist(prop_data, color=self.cc_colors(ds, 2),
                        histtype='step', fill=True,
                        density=False, bins=10, log=True,
                        range=xlim, alpha=.9, stacked=True)
                ax.hist(plot_data[0], color=self.cc_colors(ds, 0),
                        histtype='step', fill=True,
                        density=False, bins=10, log=True,
                        range=xlim, alpha=.9, stacked=True)
            else:
                plot_data = [prop_data]
                colors = [self.cc_colors(ds, 0)]
                print(ds, prop, np.min(plot_data[0]), np.max(plot_data[0]))
                ax.hist(plot_data, color=colors,
                        histtype='step',
                        density=False, bins=20, log=True,
                        range=xlim, alpha=.9, stacked=True)
            # set limits
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            # ax.set_yscale('log')
            ax.set_yticks([1e2, 1e4, 1e6])
            xmin, xmax = xlim
            ax.set_xticks(np.linspace(xmin, xmax, 5))
            ticks_str = ['%.1f' % x for x in np.linspace(xmin, xmax, 5)]
            ticks_str[0] = '%i' % xmin
            ticks_str[-1] = '%i' % xmax
            ax.set_xticklabels(ticks_str)

            # axis ticks
            if ds[:2] == 'E1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
                ax.xaxis.set_ticklabels([])
            elif ds[0] == 'E':
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # axis labels
            if ds[0] == 'A':
                ax.set_xlabel(ds[1], fontsize=16, labelpad=8)
                ax.xaxis.set_label_position('top')
            if ds[1] == '5':
                ax.set_ylabel(ds[0], fontsize=16, rotation=0, va='center',
                              labelpad=12)
                ax.yaxis.set_label_position('right')

        # plt.minorticks_off()
        outfile = os.path.join(
            self.plot_path, '%s.png' % '_'.join([cctype, molset, prop]))
        plt.savefig(outfile, dpi=self.dpi)
        if self.svg:
            plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

    def sign3_error_predictors(self, sign2_universe_presence):
        from chemicalchecker.tool.adanet import AdaNet
        df = pd.DataFrame(columns=['dataset', 'sign2_count', 'algo', 'mse',
                                   'pearson', 'mae'])
        errors = dict()
        for ds in self.cc.datasets:
            # get real data
            s3 = self.cc.get_signature('sign3', 'full', ds)
            s2 = self.cc.get_signature('sign2', 'full', ds)
            s2_idxs = np.argwhere(
                np.isin(list(s3.keys), list(s2.keys), assume_unique=True)).flatten()
            ss2 = s2[:100000]
            s2_idxs = s2_idxs[:100000]
            ss3 = s3[:][s2_idxs]
            with h5py.File(sign2_universe_presence, 'r') as fh:
                x_real = fh['V'][:][s2_idxs]
            y_real = np.log10(np.expand_dims(
                np.mean(((ss2 - ss3)**2), axis=1), 1))

            row = {
                'dataset': ds,
                'sign2_count': len(x_real)
            }
            # load predictors
            eval_err_path = os.path.join(s3.model_path, 'adanet_error_eval')
            error_pred_fn = AdaNet.predict_fn(
                os.path.join(eval_err_path, 'savedmodel'))
            lr = pickle.load(
                open(os.path.join(eval_err_path, 'LinearRegression.pkl')))
            rf = pickle.load(
                open(os.path.join(eval_err_path, 'RandomForest.pkl')))

            predictions = {
                'NeuralNetwork': AdaNet.predict(x_real, error_pred_fn)[:, 0],
                'LinearRegression': lr.predict(x_real),
                'RandomForest': rf.predict(x_real)
            }
            y_flat = y_real[:, 0]

            errors[ds] = list()
            for algo, pred in predictions.items():
                errors[ds].append((algo, y_flat - pred))
                row['algo'] = algo
                row['mse'] = np.mean((y_flat - pred)**2)
                row['mae'] = np.mean(y_flat - pred)
                row['pearson'] = np.corrcoef(y_flat, pred)[0][1]
                df.loc[len(df)] = pd.Series(row)
                print(row)

        df['algo'] = df.algo.map(
            {'LinearRegression': 'LR', 'RandomForest': 'RF', 'NeuralNetwork': 'NN'})
        # sns.set_style("whitegrid")
        sns.set_context("talk")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 15), dpi=self.dpi)

        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):

            sns.barplot(x=[x[0] for x in errors[ds]], y=[x[1]
                                                         for x in errors[ds]], ax=ax)
            # ax.set_ylim(-0.15, .15)
            # ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_ylabel('')
            # ax.set_xticklabels()

            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

        # plt.tight_layout()
        filename = os.path.join(self.plot_path, "sign3_error_predictors.png")
        plt.savefig(filename, dpi=self.dpi)
        plt.close('all')
        return df

    def sign3_confidence_distribution(self):

        # sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 10), dpi=self.dpi)

        for ds, ax in zip(self.datasets, axes.flat):
            sign3 = self.cc.get_signature('sign3', 'full', ds)
            error_file = os.path.join(sign3.model_path, 'error.h5')
            with h5py.File(error_file, "r") as hf:
                keys = hf['keys'][:]
                train_log_mse = hf['log_mse_consensus'][:]
                train_log_mse_real = hf['log_mse'][:]
            # test is anything that wasn't in the confidence distribution
            test_keys = list(sign3.unique_keys - set(keys))
            test_idxs = np.where(
                np.isin(list(sign3.keys), test_keys, assume_unique=True))[0]
            train_idxs = np.where(
                ~np.isin(list(sign3.keys), test_keys, assume_unique=True))[0]

            # decide sample molecules
            s3_stddev = sign3.get_h5_dataset('stddev_norm')
            s3_intensity = sign3.get_h5_dataset('intensity_norm')
            s3_experr = sign3.get_h5_dataset('exp_error_norm')
            s3_conf = (s3_intensity * (1 - s3_stddev))**(1 / 2.)
            s3_conf_new = (s3_intensity * (1 - s3_stddev)
                           * (1 - s3_experr))**(1 / 3.)
            # s3_conf = s3.get_h5_dataset('confidence')
            pc_inte = abs(stats.pearsonr(
                s3_intensity[train_idxs], train_log_mse)[0])
            pc_stddev = abs(stats.pearsonr(
                s3_stddev[train_idxs], train_log_mse)[0])
            pc_experr = abs(stats.pearsonr(
                s3_experr[train_idxs], train_log_mse)[0])
            s3_conf_new_w = np.average(
                [s3_intensity, (1 - s3_stddev), (1 - s3_experr)], axis=0, weights=[1, 1, pc_experr])

            df = pd.DataFrame({'train': True, 'confidence': s3_conf[
                              train_idxs], 'kind': 'old'})
            df = pd.concat([ df, pd.DataFrame({'train': False, 'confidence': s3_conf[
                           test_idxs], 'kind': 'old'}) ], ignore_index=True)
            df = pd.concat([ df, pd.DataFrame({'train': True, 'confidence': s3_conf_new[
                           train_idxs], 'kind': 'new'}) ], ignore_index=True)
            df = pd.concat([ df, pd.DataFrame({'train': False, 'confidence': s3_conf_new[
                           test_idxs], 'kind': 'new'}) ], ignore_index=True)
            df = pd.concat([ df, pd.DataFrame({'train': True, 'confidence': s3_conf_new_w[
                           train_idxs], 'kind': 'test'}) ], ignore_index=True)
            df = pd.concat([ df, pd.DataFrame({'train': False, 'confidence': s3_conf_new_w[
                           test_idxs], 'kind': 'test'}) ], ignore_index=True)
            # get idx of nearest neighbors of s2
            sns.boxplot(data=df, y='confidence', x='kind', hue='train',
                        order=['old', 'new', 'test'],
                        hue_order=[True, False],
                        color=self.cc_palette([ds])[0], ax=ax,)
            # ax.set_yscale('log')
            ax.get_legend().remove()
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

            outfile = os.path.join(
                self.plot_path, 'confidence_distribution_new.png')
            plt.savefig(outfile, dpi=self.dpi)
        if self.svg:
            plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

    def sign3_test_error_distribution(self):

        from chemicalchecker.tool.adanet import AdaNet
        from chemicalchecker.util.splitter import Traintest
        from chemicalchecker.core.sign3 import subsample_x_only

        def row_wise_correlation(X, Y):
            var1 = (X.T - np.mean(X, axis=1)).T
            var2 = (Y.T - np.mean(Y, axis=1)).T
            cov = np.mean(var1 * var2, axis=1)
            return cov / (np.std(X, axis=1) * np.std(Y, axis=1))

        def mask_exclude(idxs, x_data, y_data):
            x_data_transf = np.copy(x_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            not_nan = np.isfinite(x_data_transf).any(axis=1)
            x_data_transf = x_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x_data_transf, y_data_transf

        # sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 10), dpi=self.dpi)
        all_dss = list(self.datasets)
        for ds, ax in zip(all_dss, axes.flat):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            # filter most correlated spaces
            ds_corr = s3.get_h5_dataset('datasets_correlation')
            corr_spaces = np.array(list(self.cc.datasets))[
                ds_corr > .9].tolist()
            self.__log.info('masking %s' % str(corr_spaces))
            if ds in corr_spaces:
                dss = corr_spaces
            else:
                dss = [ds]
            idxs = [all_dss.index(d) for d in dss]
            mask_fn = partial(mask_exclude, idxs)
            # load DNN
            predict_fn = AdaNet.predict_fn(os.path.join(
                s3.model_path, 'adanet_eval', 'savedmodel'))
            # load X Y data
            traintest_file = os.path.join(s3.model_path, 'traintest.h5')
            traintest = Traintest(traintest_file, 'test')
            traintest.open()
            x_test, y_test = traintest.get_xy(0, 1000)
            y_pred_nomask = AdaNet.predict(x_test, predict_fn)
            x_test, y_test = mask_fn(x_test, y_test)
            traintest.close()

            # get the predictions and consensus
            self.__log.info('prediction consensus 5')
            y_pred, samples = AdaNet.predict(x_test, predict_fn,
                                             subsample_x_only,
                                             consensus=True,
                                             samples=5)
            y_pred_consensus = np.mean(samples, axis=1)
            self.__log.info('prediction consensus 10')
            y_pred, samples = AdaNet.predict(x_test, predict_fn,
                                             subsample_x_only,
                                             consensus=True,
                                             samples=10)
            y_pred_consensus_10 = np.mean(samples, axis=1)
            self.__log.info('prediction consensus 20')
            y_pred, samples = AdaNet.predict(x_test, predict_fn,
                                             subsample_x_only,
                                             consensus=True,
                                             samples=20)
            y_pred_consensus_20 = np.mean(samples, axis=1)
            self.__log.info('plotting')
            mse = np.mean((y_pred - y_test)**2, axis=1)
            mse_nomask = np.mean((y_pred_nomask - y_test)**2, axis=1)
            mse_consensus = np.mean((y_pred_consensus - y_test)**2, axis=1)
            mse_consensus_10 = np.mean(
                (y_pred_consensus_10 - y_test)**2, axis=1)
            mse_consensus_20 = np.mean(
                (y_pred_consensus_20 - y_test)**2, axis=1)

            sns.distplot(np.log10(mse_nomask), ax=ax,
                         color='orange', label='cons. 1 nomask')
            sns.distplot(np.log10(mse), ax=ax,
                         color='red', label='cons. 1')
            sns.distplot(np.log10(mse_consensus), ax=ax,
                         color='green', label='cons. 5')
            sns.distplot(np.log10(mse_consensus_10), ax=ax,
                         color='blue', label='cons. 10')
            sns.distplot(np.log10(mse_consensus_20), ax=ax,
                         color='purple', label='cons. 20')
            '''
            corr_test = row_wise_correlation(y_pred_test, y_true_test)
            sns.distplot(corr_test, ax=ax, bins=20, hist_kws={'range': (0, 1)},
                         color='grey', label='%s mols.' % y_pred_test.shape[0])
            corr_test_comp = row_wise_correlation(y_pred_test.T, y_true_test.T)
            sns.distplot(corr_test_comp, ax=ax,
                         color=self.cc_palette([ds])[0], label='128 comp.')
            '''
            # err_test = np.mean((y_pred_test - y_true_test)**2, axis=1)
            # pc_corr_err = stats.pearsonr(corr_test, err_test)[0]
            # ax.text(0.05, 0.85, "p: {:.2f}".format(pc_corr_err),
            #        transform=ax.transAxes, size=10)
            # ax.set_xlim(0, 1)
            ax.legend(prop={'size': 3})
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

            outfile = os.path.join(
                self.plot_path, 'sign3_test_error_distribution.png')
            plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign3_correlation_distribution(self):

        from chemicalchecker.tool.adanet import AdaNet
        from chemicalchecker.util.splitter import Traintest

        def row_wise_correlation(X, Y):
            var1 = (X.T - np.mean(X, axis=1)).T
            var2 = (Y.T - np.mean(Y, axis=1)).T
            cov = np.mean(var1 * var2, axis=1)
            return cov / (np.std(X, axis=1) * np.std(Y, axis=1))

        def mask_exclude(idxs, x_data, y_data):
            x_data_transf = np.copy(x_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            not_nan = np.isfinite(x_data_transf).any(axis=1)
            x_data_transf = x_data_transf[not_nan]
            y_data_transf = y_data[not_nan]
            return x_data_transf, y_data_transf

        # sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=False, sharex=True,
                                 figsize=(10, 10), dpi=self.dpi)
        all_dss = list(self.datasets)
        for ds, ax in zip(all_dss, axes.flat):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            # filter most correlated spaces
            ds_corr = s3.get_h5_dataset('datasets_correlation')
            self.__log.info(str(zip(list(self.datasets), list(ds_corr))))
            # load X Y data
            traintest_file = os.path.join(s3.model_path, 'traintest.h5')
            traintest = Traintest(traintest_file, 'test')
            traintest.open()
            x_test, y_test = traintest.get_xy(0, 1000)
            traintest.close()
            # load DNN
            predict_fn = AdaNet.predict_fn(os.path.join(
                s3.model_path, 'adanet_eval', 'savedmodel'))
            # check various correlations thresholds
            colors = ['firebrick', 'gold', 'forestgreen']
            for corr_thr, color in zip([.7, .9, 1.0], colors):
                corr_spaces = np.array(list(self.datasets))[
                    ds_corr > corr_thr].tolist()
                self.__log.info('masking %s' % str(corr_spaces))
                idxs = [all_dss.index(d) for d in corr_spaces]
                x_thr, y_true = mask_exclude(idxs, x_test, y_test)
                y_pred = AdaNet.predict(x_thr, predict_fn)

                corr_test_comp = row_wise_correlation(y_pred.T, y_true.T)
                self.__log.info('%.2f N(%.2f,%.2f)' % (
                    corr_thr, np.mean(corr_test_comp), np.std(corr_test_comp)))
                sns.distplot(corr_test_comp, ax=ax,
                             color=color, label='%.2f' % corr_thr)

            ax.legend(prop={'size': 6})
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

            outfile = os.path.join(
                self.plot_path, 'sign3_correlation_distribution.png')
            plt.savefig(outfile, dpi=self.dpi)
            if self.svg:
                plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

    def sign3_test_distribution(self, cctype='sign4', limit=10000,
                                options=[['Pearson'], [True], [False]]):

        def row_wise_correlation(X, Y):
            var1 = (X.T - np.mean(X, axis=1)).T
            var2 = (Y.T - np.mean(Y, axis=1)).T
            cov = np.mean(var1 * var2, axis=1)
            return cov / (np.std(X, axis=1) * np.std(Y, axis=1))

        df = pd.DataFrame(
            columns=['dataset', 'scaled', 'comp_wise', 'metric', 'value'])
        all_dss = list(self.datasets)
        #all_dfs = list()
        for ds in all_dss:
            sign = self.cc.get_signature(cctype, 'full', ds)
            pred_file = os.path.join(
                sign.model_path, 'siamese_eval', 'plot_preds.pkl')

            if not os.path.isfile(pred_file):
                self.__log.warning('%s not found!' % pred_file)
                continue

            if options is None:
                options = [
                    ['log10MSE', 'R2', 'Pearson', 'MCC'],
                    [True, False],
                    [True, False]
                ]
            preds = pickle.load(open(pred_file, 'rb'))
            for metric, scaled, comp_wise in itertools.product(*options):
                y_true = preds['test']['ONLY-SELF']
                y_pred = preds['test']['NOT-SELF']
                if comp_wise:
                    y_true = y_true.T
                    y_pred = y_pred.T
                if scaled:
                    y_true = robust_scale(y_true)
                    y_pred = robust_scale(y_pred)
                if metric == 'log10MSE':
                    values = np.log10(np.mean((y_true - y_pred)**2, axis=1))
                elif metric == 'R2':
                    values = r2_score(y_true, y_pred, multioutput='raw_values')
                elif metric == 'Pearson':
                    values = row_wise_correlation(y_true, y_pred)
                elif metric == 'MCC':
                    y_true = y_true > 0
                    y_pred = y_pred > 0
                    values = [matthews_corrcoef(
                        y_true[i], y_pred[i]) for i in range(len(y_true))]
                _df = pd.DataFrame(
                    dict(dataset=ds, scaled=scaled, comp_wise=comp_wise,
                         metric=metric, value=values))
                df = pd.concat([ df, _df ], ignore_index=True)
                #all_dfs.append(_df)
        #df = pd.concat(all_dfs)

        # sns.set_style("ticks")
        # sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        if options is None:
            options = [
                ['log10MSE', 'R2', 'Pearson', 'MCC'],
                [True, False],
                [True, False]
            ]
        for metric, scaled, comp_wise in itertools.product(*options):
            odf = df[(df.scaled == scaled) & (df.comp_wise == comp_wise) & (
                df.metric == metric)]
            xmin = np.floor(np.percentile(odf.value, 5))
            xmax = np.ceil(np.percentile(odf.value, 95))
            fig, axes = plt.subplots(
                26, 1, sharex=True, figsize=(3, 10), dpi=self.dpi)
            fig.subplots_adjust(left=0.05, right=.95, bottom=0.08,
                                top=1, wspace=0, hspace=-.3)
            for idx, (ds, ax) in enumerate(zip(all_dss, axes.flat)):
                color = self.cc_colors(ds, idx % 2)
                color2 = self.cc_colors(ds, (idx % 2) + 1)
                values = odf[(odf.dataset == ds)].value.tolist()
                sns.kdeplot(values, ax=ax, clip_on=True, shade=True,
                            alpha=1, lw=0,  bw=.15, color=color)
                sns.kdeplot(values, ax=ax, clip_on=True,
                            color=color2, lw=2, bw=.15)

                ax.axhline(y=0, lw=2, clip_on=True, color=color)
                ax.set_xlim(xmin, xmax)
                ax.tick_params(axis='x', colors=color)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.patch.set_alpha(0)
                sns.despine(ax=ax, bottom=True, left=True, trim=True)
                ax.grid(False)
            ax = axes.flat[-1]
            ax.set_yticks([])
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(np.linspace(xmin, xmax, 5))
            ax.set_xticklabels(
                ['%.1f' % x for x in np.linspace(xmin, xmax, 5)])
            ax.grid(False)
            # ax.set_xticklabels(['0','0.5','1'])

            xlabel = metric
            if comp_wise:
                xlabel += ' Comp.'
            else:
                xlabel += ' Mol.'
            if scaled:
                xlabel += ' scaled'
            ax.set_xlabel(xlabel,
                          fontdict=dict(name='Arial', size=16))
            ax.tick_params(labelsize=14)
            ax.patch.set_alpha(0)
            sns.despine(ax=ax, bottom=False, left=True, trim=True)
            fname = 'sign3_test_dist_%s' % metric
            if comp_wise:
                fname += '_comp'
            if scaled:
                fname += '_scaled'
            print(fname)
            print(odf.value.describe())
            outfile = os.path.join(self.plot_path, fname + '.png')
            plt.savefig(outfile, dpi=self.dpi)
            if self.svg:
                plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
            plt.close('all')

    def sign3_mfp_predictor(self):

        # sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey=True)

        for ds, ax in zip(self.datasets, axes.flat):
            try:
                s3 = self.cc.get_signature('sign3', 'full', ds)
            except Exception:
                continue
            if not os.path.isfile(s3.data_path):
                continue
            stat_file = os.path.join(s3.model_path,
                                     'adanet_sign0_A1.001_eval',
                                     'stats_sign0_A1.001_eval.pkl')
            df = pd.read_pickle(stat_file)
            df['component_cat'] = pd.cut(
                df.component,
                bins=[-1, 127, 128, 129, 130, 131, 132],
                labels=['signature', 'stddev', 'intensity', 'exp_error',
                        'novelty', 'confidence'])
            # get idx of nearest neighbors of s2
            sns.barplot(x='component_cat', y='mse', data=df, hue='split',
                        hue_order=['train', 'test'],
                        ax=ax, color=self.cc_palette([ds])[0])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(prop={'size': 6})
            ax.get_legend().remove()
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

        f.text(0.06, 0.5, 'MSE', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_mfp_predictor.png')
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign3_mfp_confidence_predictor(self):

        # sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey=True)

        for ds, ax in zip(self.datasets, axes.flat):
            try:
                s3 = self.cc.get_signature('sign3', 'full', ds)
            except Exception:
                continue
            if not os.path.isfile(s3.data_path):
                continue
            stat_file = os.path.join(s3.model_path,
                                     'adanet_sign0_A1.001_conf_eval',
                                     'stats_sign0_A1.001_conf_eval.pkl')
            df = pd.read_pickle(stat_file)
            df['component_cat'] = df.component.astype('category')
            df['component_cat'] = df['component_cat'].cat.rename_categories(
                ['stddev', 'intensity', 'exp_error', 'novelty', 'confidence'])

            # get idx of nearest neighbors of s2
            sns.barplot(x='component_cat', y='mse', data=df, hue='split',
                        hue_order=['train', 'test'],
                        ax=ax, color=self.cc_palette([ds])[0])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(prop={'size': 6})
            ax.get_legend().remove()
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])

        f.text(0.06, 0.5, 'MSE', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_mfp_confidence_predictor.png')
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    @staticmethod
    def quick_gaussian_kde(x, y, limit=1000):
        xl = x[:limit]
        yl = y[:limit]
        xy = np.vstack([xl, yl])
        try:
            c = gaussian_kde(xy)(xy)
        except Exception as ex:
            MultiPlot.__log.warning('Could not compute KDE: %s' % str(ex))
            c = np.arange(len(xy))
        order = c.argsort()
        return xl, yl, c, order

    def sign3_confidence_summary(self, limit=5000):

        from chemicalchecker.core.signature_data import DataSignature

        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        gs = fig.add_gridspec(5, 5, wspace=0.1, hspace=0.1)
        plt.subplots_adjust(left=0.08, right=.95, bottom=0.08, top=.95)
        axes = list()
        for row, col in itertools.product(range(5), range(5)):
            axes.append(fig.add_subplot(gs[row, col]))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False,
                        bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Correlation", labelpad=25, size=18)
        plt.ylabel("Applicability", labelpad=25, size=18)

        for ds, ax in zip(self.datasets, axes):
            sign = self.cc.get_signature('sign3', 'full', ds)

            # get data
            confidence_path = os.path.join(sign.model_path, 'confidence_eval')
            known_dist = DataSignature(os.path.join(
                confidence_path, 'data.h5'))
            correlation = known_dist.get_h5_dataset('y_test')
            conf_feats = known_dist.get_h5_dataset('x_test')
            applicability = conf_feats[:, 0]
            robustness = conf_feats[:, 1]
            prior = conf_feats[:, 2]
            prior_sign = conf_feats[:, 3]
            intensity = conf_feats[:, 4]

            # get confidence
            confidence_file = os.path.join(confidence_path, 'confidence.pkl')
            calibration_file = os.path.join(confidence_path, 'calibration.pkl')
            conf_mdl = (pickle.load(open(confidence_file, 'rb')),
                        pickle.load(open(calibration_file, 'rb')))
            # and estimate confidence
            conf_feats = np.vstack(
                [applicability, robustness, prior, prior_sign, intensity]).T
            conf_estimate = conf_mdl[0].predict(conf_feats)
            confidence = conf_mdl[1].predict(np.expand_dims(conf_estimate, 1))

            # compute pearson rho
            rhos = dict()
            rho_confidence = stats.pearsonr(correlation, confidence)[0]
            rhos['d'] = abs(stats.pearsonr(correlation, applicability)[0])
            rhos['r'] = abs(stats.pearsonr(correlation, robustness)[0])
            rhos['p'] = abs(stats.pearsonr(correlation, prior)[0])
            rhos['s'] = abs(stats.pearsonr(correlation, prior_sign)[0])
            rhos['i'] = abs(stats.pearsonr(correlation, intensity)[0])

            # scatter gaussian
            x, y, c, order = self.quick_gaussian_kde(
                correlation, confidence, limit)
            colors = [self.cc_colors(ds, 2), self.cc_colors(ds, 0)]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                '', colors)
            ax.scatter(x[order], y[order], c=c[order],
                       cmap=cmap, s=15, edgecolor='', alpha=.9)
            ax.text(0.05, 0.85, r"$\rho$: {:.2f}".format(rho_confidence),
                    transform=ax.transAxes, name='Arial', size=14,
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylabel('Applicability')
            ax.set_ylabel('')
            ax.set_xlabel('Correlation')
            ax.set_xlabel('')

            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".9")

            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
            ax.tick_params(labelsize=14, direction='inout')

            # axis ticks
            if ds[:2] == 'E1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
                ax.xaxis.set_ticklabels([])
            elif ds[0] == 'E':
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

            # axis labels
            if ds[0] == 'A':
                ax.set_xlabel(ds[1], fontsize=18, labelpad=15)
                ax.xaxis.set_label_position('top')
            if ds[1] == '5':
                ax.set_ylabel(ds[0], fontsize=18, rotation=0, va='center',
                              labelpad=15)
                ax.yaxis.set_label_position('right')

            # pies
            wp = {'linewidth': 0, 'antialiased': True}
            colors = [self.cc_colors(ds, 1), 'lightgrey']
            i = 0
            for name, rho in rhos.items():
                # [x0, y0, width, height]
                bounds = [0.05 + (0.18 * i), 0.05, 0.18, 0.18]
                i += 1
                inset_ax = ax.inset_axes(bounds)
                inset_ax.pie([abs(rho), 1 - abs(rho)], wedgeprops=wp,
                             counterclock=False, startangle=90, colors=colors)
                inset_ax.pie([1.0], radius=0.5, colors=[
                             'white'], wedgeprops=wp)
                inset_ax.text(0.5, 1.2, name, ha='center', va='center',
                              transform=inset_ax.transAxes, name='Arial',
                              style='italic', size=12)

            outfile = os.path.join(self.plot_path,
                                   'sign3_confidence_summary.png')
            plt.savefig(outfile, dpi=self.dpi)
            if self.svg:
                plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.savefig(outfile, dpi=self.dpi)
        if self.svg:
            plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
        plt.close('all')

    def sign3_novel_confidence_distribution(self):

        # sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey=True)

        for ds, ax in zip(self.datasets, axes.flat):
            try:
                s3 = self.cc.get_signature('sign3', 'full', ds)
            except Exception:
                continue
            if not os.path.isfile(s3.data_path):
                continue
            # get novelty and confidence
            # nov = s3.get_h5_dataset('novelty')
            out = s3.get_h5_dataset('outlier')
            conf = s3.get_h5_dataset('confidence')
            # get really novel molecules
            # min_known = min(nov[out == 0])
            nov_confs = conf[out == -1]
            print(ds, len(nov_confs))
            sns.distplot(nov_confs, color=self.cc_palette([ds])[0],
                         kde=False, norm_hist=False, ax=ax, bins=20,
                         hist_kws={'range': (0, 1)})
            ax.set_xlim(0, 1)
            ax.set_yscale('log')
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.grid(axis='x', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
            ax.text(0.05, 0.85, "%i" % len(nov_confs),
                    transform=ax.transAxes, size=8)
        f.text(0.5, 0.04, 'Confidence', ha='center', va='center')
        f.text(0.06, 0.5, 'Novel Molecules', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_novel_confidence_distribution.png')
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')

    def sign3_examplary_test_correlation(self, limit=1000,
                                         molecules=[
                                             'MBUVEWMHONZEQD-UHFFFAOYSA-N'],
                                         examplary_ds=['B1.001', 'D1.001', 'E4.001']):

        from chemicalchecker.core import DataSignature

        def mask_keep(idxs, x1_data):
            # we will fill an array of NaN with values we want to keep
            x1_data_transf = np.zeros_like(x1_data, dtype=float) * np.nan
            for idx in idxs:
                # copy column from original data
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = x1_data[:, col_slice]
            # keep rows containing at least one not-NaN value
            return x1_data_transf

        def mask_exclude(idxs, x1_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            # drop rows that only contain NaNs
            return x1_data_transf

        def row_wise_correlation(X, Y):
            var1 = (X.T - np.mean(X, axis=1)).T
            var2 = (Y.T - np.mean(Y, axis=1)).T
            cov = np.mean(var1 * var2, axis=1)
            return cov / (np.std(X, axis=1) * np.std(Y, axis=1))

        def plot_molecule(ax, smiles, size=6):
            from rdkit import Chem
            from rdkit.Chem import Draw
            figure = Draw.MolToMPL(Chem.MolFromSmiles(smiles))
            # rdkit only plot to new figure, so copy over to my axq
            for child in figure.axes[0].get_children():
                if isinstance(child, matplotlib.lines.Line2D):
                    ax.plot(*child.get_data(), c=child.get_color())
                if isinstance(child, matplotlib.text.Annotation):
                    ax.text(*child.get_position(), s=child.get_text(),
                            color=child.get_color(), ha=child.get_ha(),
                            va=child.get_va(), family=child.get_family(),
                            size=size,
                            bbox={'facecolor': 'white',
                                  'boxstyle': 'circle,pad=0.2'})
            plt.close(figure)

        correlations = dict()
        true_pred = dict()
        for ds in examplary_ds:
            ds_idx = np.argwhere(np.isin(self.datasets, ds)).flatten()
            s2 = self.cc.get_signature('sign2', 'full', ds)
            s4 = self.cc.get_signature('sign3', 'full', ds)
            traintest_file = os.path.join(s4.model_path, 'traintest_eval.h5')
            traintest = DataSignature(traintest_file)
            inks = traintest.get_h5_dataset('keys_train')[:100000]
            inks = np.sort(inks)
            test_mask = np.isin(list(s2.keys), list(inks),
                                assume_unique=True)
            sign2_matrix = os.path.join(s4.model_path, 'train.h5')
            X = DataSignature(sign2_matrix)
            feat = X.get_h5_dataset('x', mask=test_mask)
            predict_fn = s4.get_predict_fn(smiles=False, model='siamese_eval')
            y_true = predict_fn(mask_keep(ds_idx, feat))
            y_pred = predict_fn(mask_exclude(ds_idx, feat))

            true_pred[ds] = dict()
            for i, ink in enumerate(inks):
                tp = (y_true[i], y_pred[i])
                true_pred[ds][ink] = tp
            ink_corrs = list(zip(inks, row_wise_correlation(y_true, y_pred)))
            correlations[ds] = dict(ink_corrs)

        if molecules is None:
            mols = list()
            for k, v in true_pred.items():
                mols.append(set(v.keys()))
            shared_inks = set.intersection(*mols)
            print(shared_inks)
        else:
            shared_inks = molecules

        for mol in tqdm(shared_inks):
            # sns.set_style("whitegrid")
            # sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
            fig = plt.figure(constrained_layout=True, figsize=(4, 4))
            fig.set_constrained_layout_pads(w_pad=0., h_pad=0.,
                                            hspace=0., wspace=0.)
            gs = fig.add_gridspec(2, 1)
            gs.set_height_ratios((1, 1))
            # gs.set_height_ratios((1, 2, 2, 2))
            '''
            fig.text(0.5, 0.02, 'Actual Signature',
                     ha='center', va='center',
                     name='Arial', size=16)
            fig.text(0.04, 0.45, 'Predicted', ha='center',
                     va='center', rotation='vertical',
                     name='Arial', size=16)
            '''
            # plot molecule
            ax_mol = fig.add_subplot(gs[0])
            # get smiles
            converter = Converter()
            smiles = converter.inchi_to_smiles(
                converter.inchikey_to_inchi(mol)[0]['standardinchi'])
            plot_molecule(ax_mol, smiles, size=8)
            # fix placement
            ax_mol.set_axis_off()
            l, b, w, h = ax_mol.get_position().bounds
            new_w = w + 0.05
            new_h = h + 0.05
            ax_mol.set_position([0.5 - (new_w / 2.), b + 0.05, new_w, new_h])
            ax_mol.axis('equal')
            gss_ds = gs[1].subgridspec(1, len(examplary_ds))
            mccs = list()
            for idx, (ds, sub) in enumerate(zip(examplary_ds, gss_ds)):
                gs_ds = sub.subgridspec(2, 2, wspace=0.0, hspace=0.0)
                gs_ds.set_height_ratios((1, 5))
                gs_ds.set_width_ratios((5, 1))
                ax_main = fig.add_subplot(gs_ds[1, 0])
                ax_top = fig.add_subplot(gs_ds[0, 0], sharex=ax_main)
                ax_top.text(-0.2, 0.2, "%s" % ds[:2],
                            color='black',
                            transform=ax_top.transAxes,
                            name='Arial', size=14, weight='bold')
                ax_top.set_axis_off()
                ax_right = fig.add_subplot(gs_ds[1, 1], sharey=ax_main)
                ax_right.set_axis_off()

                true, pred = true_pred[ds][mol]
                T = true > 0
                P = pred > 0
                from scipy.signal import find_peaks
                kde_range = np.linspace(-1, 1, 1000)
                peaks_true_i, _ = find_peaks(gaussian_kde(true)(kde_range))
                peaks_pred_i, _ = find_peaks(gaussian_kde(pred)(kde_range))
                peaks_true = kde_range[peaks_true_i]
                peaks_pred = kde_range[peaks_pred_i]

                coords = np.array(
                    [[max(peaks_true), max(peaks_pred)],
                     [min(peaks_true), max(peaks_pred)],
                     [min(peaks_true), min(peaks_pred)],
                     [max(peaks_true), min(peaks_pred)]])
                x_range = (min(peaks_true) - 0.1, max(peaks_true) + 0.1)
                y_range = (min(peaks_pred) - 0.1, max(peaks_pred) + 0.1)
                ax_main.set_xlim(x_range)
                ax_main.set_ylim(y_range)
                s = np.array(
                    [sum(T & P), sum(~T & P), sum(~T & ~P), sum(T & ~P)])
                ax_main.scatter(coords[:, 0], coords[:, 1], s=s * 10,
                                color=self.cc_colors(ds, 1),
                                edgecolor="black", lw=0.8)

                # sns.despine(ax=ax_main, offset=3, trim=True)

                ax_main.set_xlabel('Actual', size=14)
                ax_main.set_ylabel('Inferred', size=14)
                if idx != 0:
                    ax_main.set_ylabel(' ')
                ax_main.set_yticks([0])
                ax_main.set_yticklabels([])
                ax_main.set_xticks([0])
                ax_main.set_xticklabels([])
                ax_main.tick_params(labelsize=14, direction='inout')
                # ax_main.set_aspect('equal')
                mcc = matthews_corrcoef(T, P)
                mccs.append(mcc)
                ax_main.text(0.5, 0.5, r"$MCC$: {:.2f}".format(mcc),
                             transform=ax_main.transAxes, name='Arial',
                             ha='center', va='center',
                             size=12, bbox=dict(facecolor='white', alpha=0.8))

                sns.distplot(true, ax=ax_top, kde=True,
                             hist=False, kde_kws=dict(shade=True, bw=.2),
                             color=self.cc_colors(ds, 1))
                sns.distplot(pred, ax=ax_right, vertical=True,  kde=True,
                             hist=False, kde_kws=dict(shade=True, bw=.2),
                             color=self.cc_colors(ds, 1))
                '''
                if idx == 0:
                    ax_main.set_xlabel(' ')
                if idx == 1:
                    ax_main.set_ylabel(' ')
                if idx == 2:
                    ax_main.set_ylabel(' ')
                    ax_main.set_xlabel(' ')
                '''
            # plt.tight_layout()
            spaces = '_'.join(['%s-%.1f' % (ds[:2], m)
                               for ds, m in zip(examplary_ds, mccs)])
            outfile = os.path.join(
                self.plot_path, 'sign3_%s_%s.png' % (spaces, mol))
            plt.savefig(outfile, dpi=self.dpi)
            if self.svg:
                plt.savefig(outfile.replace('.png', '.svg'), dpi=self.dpi)
            plt.close('all')

    def cctype_validation_comparison(self, cctype1='sign4', cctype2='sign2', valtype='moa'):

        pklfile = os.path.join(
            self.plot_path, '%s_%s_%s_comparison.pkl' % (cctype1, cctype2, valtype))
        if not os.path.isfile(pklfile):
            data = dict()
            for ds in self.datasets:
                if 'dataset' not in data:
                    data['dataset'] = list()
                data['dataset'].append(ds[:2])
                # sign2
                s2 = self.cc.get_signature(cctype2, 'full', ds)
                stat_file = os.path.join(
                    s2.stats_path, 'validation_stats.json')
                if not os.path.isfile(stat_file):
                    s2.validate()
                stats = json.load(open(stat_file, 'r'))
                for k, v in stats.items():
                    if k + '_%s' % cctype2 not in data:
                        data[k + '_%s' % cctype2] = list()
                    data[k + '_%s' % cctype2].append(v)
                # sign4
                s3 = self.cc.get_signature(cctype1, 'full', ds)
                stat_file = os.path.join(
                    s3.stats_path, 'validation_stats.json')
                if not os.path.isfile(stat_file):
                    s3.validate()
                stats = json.load(open(stat_file, 'r'))
                for k, v in stats.items():
                    if k + '_0.0' not in data:
                        data[k + '_0.0'] = list()
                    data[k + '_0.0'].append(v)
                # other confidences thresholds
                for thr in np.arange(0.1, 0.9, 0.1):
                    s3_conf = self.cc.get_signature(
                        cctype1, 'conf%.1f' % thr, ds)
                    if not os.path.isfile(s3_conf.data_path):
                        conf_mask = s3.get_h5_dataset('confidence') > thr
                        s3.make_filtered_copy(s3_conf.data_path, conf_mask)
                    stat_file = os.path.join(s3_conf.stats_path,
                                             'validation_stats.json')
                    if not os.path.isfile(stat_file):
                        s3_conf.validate()
                    stats = json.load(open(stat_file, 'r'))
                    for k, v in stats.items():
                        if k + '_%.1f' % thr not in data:
                            data[k + '_%.1f' % thr] = list()
                        data[k + '_%.1f' % thr].append(v)

            df = pd.DataFrame(data)
            df = df.infer_objects()
            df.sort_values("dataset", ascending=False, inplace=True)
            df.to_pickle(pklfile)

        def gradient_arrow(ax, start, end, xs=None, cmap="plasma", head=None, n=50, lw=3):
            # Arrow shaft: LineCollection
            if xs is None:
                x = np.linspace(start[0], end[0], n)
            else:
                x = xs
                n = len(xs)
            cmap = plt.get_cmap(cmap, n)
            y = np.linspace(start[1], end[1], n)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, linewidth=lw)
            lc.set_array(np.linspace(0, 1, n))
            ax.add_collection(lc)
            # Arrow head: Triangle
            tricoords = [(0, -0.4), (0.5, 0), (0, 0.4), (0, -0.4)]
            angle = np.arctan2(end[1] - start[1], end[0] - start[0])
            rot = matplotlib.transforms.Affine2D().rotate(angle)
            tricoords2 = rot.transform(tricoords)
            tri = matplotlib.path.Path(tricoords2, closed=True)
            if head is None:
                if xs is None:
                    head = cmap(n)
                else:
                    head = cmap(xs.index(xs[-1]))
            ax.scatter(end[0], end[1], c=head, s=(2 * lw)**2,
                       marker=tri, cmap=cmap, vmin=0)
            ax.autoscale_view()

        df = pd.read_pickle(pklfile)
        # sns.set_style("ticks")
        # sns.set_style({'font.family': 'sans-serif',  'font.serif': ['Arial']})

        fig = plt.figure(figsize=(3, 10))
        plt.subplots_adjust(left=0.14, right=0.96, bottom=0.01,
                            top=0.92, hspace=0.1)

        gs = fig.add_gridspec(2, 1)
        gs.set_height_ratios((30, 1))
        gs_ds = gs[0].subgridspec(1, 2, wspace=0.1, hspace=0.0)
        ax_cov = fig.add_subplot(gs_ds[0])
        ax_roc = fig.add_subplot(gs_ds[1])

        for ds in self.datasets:
            y = 25 - self.datasets.index(ds) - 1
            start = df[df.dataset == ds[:2]]['%s_cov_%s' % (valtype, cctype2)].tolist()[
                0]
            end = df[df.dataset == ds[:2]]['%s_cov_0.0' % valtype].tolist()[0]
            js = [cctype2] + ['%.1f' %
                              f for f in reversed(np.arange(0.0, 0.9, 0.1))]
            # js = ['sign2','0.5','0.0']
            covs = [df[df.dataset == ds[:2]]['%s_cov_%s' % (valtype, j)].tolist()[0]
                    for j in js]
            cmap = plt.get_cmap("plasma", len(covs))
            gradient_arrow(ax_cov, (start, y), (end, y), xs=covs, lw=7)

            start = df[df.dataset == ds[:2]]['%s_auc_%s' % (valtype, cctype2)].tolist()[
                0]
            aucs = [df[df.dataset == ds[:2]]['%s_auc_%s' % (valtype, j)].tolist()[0]
                    for j in js]
            end = aucs[np.argmax(covs)]
            cmap = plt.get_cmap("plasma", len(covs))
            ax_roc.scatter(start, y, color=cmap(0), s=60)
            ax_roc.scatter(end, y, color=cmap(np.argmax(covs)), s=60)

        ax_cov.grid(False)
        ax_cov.set_yticks(range(0, 25))
        ax_cov.set_yticklabels(df['dataset'])
        ax_cov.set_xlim(0, 110)
        ax_cov.set_xticks([0, 100])
        ax_cov.set_xticklabels(['0', '100'])
        ax_cov.set_xlabel('Coverage', fontdict=dict(name='Arial', size=14))
        ax_cov.xaxis.set_label_position('top')
        # ax_cov.xaxis.tick_top()
        sns.despine(ax=ax_cov, left=True, bottom=True, top=False, trim=True)
        ax_cov.tick_params(left=False, labelsize=14, direction='inout',
                           bottom=False, top=True, labelbottom=False, labeltop=True)
        # set the alignment for outer ticklabels
        ticklabels = ax_cov.get_xticklabels()
        ticklabels[0].set_ha("left")
        ticklabels[-1].set_ha("right")

        ax_roc.grid(False)
        ax_roc.set_yticks([])
        ax_roc.set_yticklabels([])
        ax_roc.set_xlim(0.5, 1)
        ax_roc.set_xticks([0.5, 1])
        ax_roc.set_xticklabels(['0.5', '1'])
        ax_roc.set_xlabel('AUROC', fontdict=dict(name='Arial', size=14))
        ax_roc.xaxis.set_label_position('top')
        sns.despine(ax=ax_roc, left=True, bottom=True, top=False, trim=True)
        ax_roc.tick_params(left=False, labelsize=14, direction='inout',
                           bottom=False, top=True, labelbottom=False, labeltop=True)
        # set the alignment for outer ticklabels
        ticklabels = ax_roc.get_xticklabels()
        ticklabels[0].set_ha("left")
        ticklabels[-1].set_ha("right")

        ax_cbar = fig.add_subplot(gs[1])
        cbar = matplotlib.colorbar.ColorbarBase(
            ax_cbar, cmap=cmap, orientation='horizontal',
            ticklocation='top')
        cbar.ax.set_xlabel('Confidence filter',
                           fontdict=dict(name='Arial', size=14))
        cbar.ax.tick_params(labelsize=14, )
        cbar.set_ticks([1, .8, .6, .4, .2, .0])
        cbar.set_ticklabels(
            list(reversed(['1', '0.8', '0.6', '0.4', '0.2', '0'])))

        outfile = os.path.join(
            self.plot_path, '%s_%s_%s_comparison.png' % (cctype1, cctype2, valtype))
        plt.savefig(outfile, dpi=self.dpi)
        plt.close('all')
