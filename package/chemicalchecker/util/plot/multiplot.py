"""Utility for plotting Chemical Checker data."""

import os
import h5py
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import interpolate
from scipy import stats
from functools import partial
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from chemicalchecker.util import logged
from chemicalchecker.util.decomposition import dataset_correlation


@logged
class MultiPlot():
    """Produce Chemical Checker plots using multiple datasets."""

    def __init__(self, chemchecker, plot_path, limit_dataset=None):
        """Initialize a MultiPlot object.

        Produce plots integrating data from multiple datasets.

        Args:
            chemchecker(str): A Chemical Checker instance.
            plot_path(str): Destination folder for plot images.
        """
        if not os.path.isdir(plot_path):
            raise Exception("Folder to save plots does not exist")
        self.__log.debug('Plots will be saved to %s', plot_path)
        self.plot_path = plot_path
        self.cc = chemchecker
        if not limit_dataset:
            self.datasets = list(self.cc.datasets)
        else:
            self.datasets = limit_dataset

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

    @staticmethod
    def cc_colors(coord, lighness=0):
        colors = {
            'A': ['#EA5A49', '#EE7B6D', '#F7BDB6'],
            'B': ['#B16BA8', '#C189B9', '#D0A6CB'],
            'C': ['#5A72B5', '#7B8EC4', '#9CAAD3'],
            'D': ['#7CAF2A', '#96BF55', '#B0CF7F'],
            'E': ['#F39426', '#F5A951', '#F8BF7D']}
        return colors[coord[:1]][lighness]

    def cmap_discretize(self, cmap, N):
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
                           colors_rgba[i, ki]) for i in xrange(N + 1)]
        # Return colormap object.
        return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

    def sign_adanet_stats(self, ctype, metric=None, compare=None):
        # read stats fields
        sign = self.cc.get_signature(ctype, 'reference', 'E5.001')
        stat_file = os.path.join(sign.model_path, 'adanet', 'stats.pkl')
        df = pd.read_pickle(stat_file)
        # merge all stats to pandas
        df = pd.DataFrame(columns=['coordinate'] + list(df.columns))
        for ds in tqdm(self.datasets):
            sign = self.cc.get_signature(ctype, 'reference', ds)
            stat_file = os.path.join(sign.model_path, 'adanet', 'stats.pkl')
            if not os.path.isfile(stat_file):
                continue
            tmpdf = pd.read_pickle(stat_file)
            tmpdf['coordinate'] = ds
            df = df.append(tmpdf, ignore_index=True)
        df = df.infer_objects()

        outfile_csv = os.path.join(self.plot_path, 'sign2_adanet_stats.csv')
        df.to_csv(outfile_csv)
        outfile_pkl = os.path.join(self.plot_path, 'sign2_adanet_stats.pkl')
        df.to_pickle(outfile_pkl)

        if compare:
            cdf = pd.read_pickle(compare)
            cdf = cdf[cdf.algo == 'AdaNet'].copy()
            cdf['algo'] = cdf.algo.apply(lambda x: x + '_STACK')
            df = df.append(cdf, ignore_index=True)

        if metric:
            all_metrics = [metric]
        else:
            all_metrics = ['mse', 'r2', 'explained_variance', 'pearson_std',
                           'pearson_avg', 'time', 'nn_layers', 'nr_variables']
        for metric in all_metrics:
            sns.set_style("whitegrid")
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
            plt.savefig(outfile, dpi=100)
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
        plt.savefig(outfile, dpi=100)
        plt.close('all')

    def sign_feature_distribution_plot(self, cctype, molset, block_size=1000,
                                       block_nr=10, sort=False):
        sample_size = block_size * block_nr
        fig, axes = plt.subplots(25, 1, sharey=True, sharex=True,
                                 figsize=(10, 40), dpi=100)
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
            df = df.append(all_df, ignore_index=True)
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
        plt.savefig(filename, dpi=100)
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

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
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
        plt.savefig(filename, dpi=100)
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
            df = df.append(tmpdf, ignore_index=True)

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
        sns.set_style("whitegrid")
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
        plt.savefig(filename, dpi=100)
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
        plt.savefig(filename, dpi=100)
        plt.close()

        for metric in ['pearson_avg', 'time', 'r2', 'pearson_std', 'explained_variance']:
            sns.set_style("whitegrid")
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
            plt.savefig(filename, dpi=100)
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
            plt.savefig(filename, dpi=100)
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
        sns.set_style("ticks")
        matplotlib.rcParams['font.sans-serif'] = "Arial"
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
        plt.savefig(filename, dpi=100)
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
            df = df.append(tmpdf, ignore_index=True)

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
        sns.set_style("whitegrid")
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
        plt.savefig(filename, dpi=100)
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
        plt.savefig(filename, dpi=100)
        plt.close()

        for metric in ['pearson_avg', 'time', 'r2', 'pearson_std', 'explained_variance']:
            sns.set_style("whitegrid")
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
            plt.savefig(filename, dpi=100)
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
            plt.savefig(filename, dpi=100)
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
        dfs = list()
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
            dfs.append(tmpdf)
        df = pd.DataFrame(columns=list(set(cols)) +
                          ['coordinate_from', 'coordinate_to', 'train_size'])
        df = df.append(dfs, ignore_index=True)
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
            plt.savefig(filename, dpi=300)
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

        sns.set_style("whitegrid")
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
        plt.savefig(filename, dpi=300)
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
        sns.set_style("white")
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
        plt.savefig(filename, dpi=100)
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
        plt.savefig(filename, dpi=100)
        plt.close()

    def sign3_adanet_performance_all_plot(self, metric="pearson", suffix=None,
                                          stat_filename="stats_eval.pkl"):

        sns.set_style("whitegrid")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
        fig, axes = plt.subplots(25, 1, sharey=True, sharex=False,
                                 figsize=(20, 70), dpi=100)
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
        plt.savefig(filename, dpi=200)
        plt.close('all')

    def sign3_adanet_performance_overall(self, metric="pearson", suffix=None,
                                         not_self=True):

        sns.set_style("whitegrid")
        sns.set_context("talk")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=False,
                                 figsize=(10, 10), dpi=100)
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
        plt.savefig(filename, dpi=100)
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
            df = df.append(sel, ignore_index=True)

        df['from'] = df['from'].map({ds: ds[:2] for ds in self.cc.datasets})
        df = df.dropna()

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(df.pivot('from', 'to', metric), vmin=0, vmax=1,
                    linewidths=.5, square=True, cmap=cmap)
        plt.title('set: %s, metric: %s' % (split, metric))
        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "adanet_perf_heatmap_%s_%s.png" % (split, metric))
        plt.savefig(filename, dpi=100)
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

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(df.pivot('from', 'to', 'coverage'), vmin=0, vmax=1,
                    linewidths=.5, square=True, cmap=cmap)
        plt.title('Coverage')
        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "sign3_coverage_heatmap.png")
        plt.savefig(filename, dpi=100)
        plt.close('all')

    def sign3_coverage_barplot(self, sign2_coverage):
        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
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

        fig, ax = plt.subplots(1, 1, figsize=(3, 10), dpi=100)
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
        ax.set_xlabel('Molecule Coverage',
                      fontdict=dict(name='Arial', size=16))
        # ax.xaxis.tick_top()

        ax.set_xticks([0, 2, 4, 6])
        ax.set_xticklabels(
            [r'$10^{0}$', r'$10^{2}$', r'$10^{4}$', r'$10^{6}$', ])
        ax.tick_params(labelsize=14)

        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "sign3_coverage_barplot.png")
        plt.savefig(filename, dpi=200)
        plt.close('all')

    def sign3_CCA_heatmap(self, limit=10000):
        df = pd.DataFrame(columns=['from', 'to', 'CCA'])
        for i in range(len(self.datasets)):
            ds_from = self.datasets[i]
            s3_from = self.cc.get_signature('sign3', 'full', ds_from)[:limit]
            for j in range(i + 1):
                ds_to = self.datasets[j]
                s3_to = self.cc.get_signature('sign3', 'full', ds_to)[:limit]
                res = dataset_correlation(s3_from, s3_to)

                df.loc[len(df)] = pd.Series({
                    'from': ds_from[:2],
                    'to': ds_to[:2],
                    'CCA': res[0]})
                if ds_to != ds_from:
                    df.loc[len(df)] = pd.Series({
                        'from': ds_to[:2],
                        'to': ds_from[:2],
                        'CCA': res[3]})
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
        cmap = plt.cm.get_cmap('plasma_r', 5)
        sns.heatmap(df.pivot('from', 'to', 'CCA'), vmin=0, vmax=1,
                    linewidths=.5, square=True, cmap=cmap)
        plt.title('Canonical Correlation Analysis')
        plt.tight_layout()
        filename = os.path.join(
            self.plot_path, "sign3_CCA_heatmap.png")
        df.to_pickle(filename[:-3] + ".pkl")
        plt.savefig(filename, dpi=100)
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
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(15, 15), dpi=100)
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
        plt.savefig(filename, dpi=100)
        plt.close('all')
        """
        for metric in metrics:
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(5, 5, sharey=True, sharex=False,
                                     figsize=(10, 10), dpi=100)
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
            plt.savefig(filename, dpi=100)
            plt.close('all')
        """

    def sign3_neig2_jaccard(self, max_sample=2000):

        df = pd.DataFrame(columns=['dataset', 'k', 'confidence', 'jaccard'])

        for ds in self.datasets:
            try:
                s2 = self.cc.get_signature('sign2', 'full', ds)
                n2 = self.cc.get_signature('neig2', 'reference', ds)
                s3 = self.cc.get_signature('sign3', 'full', ds)
            except:
                continue
            if not os.path.isfile(s2.data_path):
                continue
            if not os.path.isfile(n2.data_path):
                continue
            if not os.path.isfile(s3.data_path):
                continue
            # decide sample molecules
            inks = s2.keys
            s3_conf = s3.get_h5_dataset('confidence')
            if len(inks) > max_sample:
                inks = list()
                s2_mask = np.isin(list(s3.keys), list(
                    s2.keys), assume_unique=True)
                s2_conf = s3_conf[s2_mask]
                # make sure we sample a bit all confidences
                conf_bins = np.arange(0, 100, 10)
                for low in conf_bins:
                    upp = low + 10
                    if low == 0:
                        mask_low = s2_conf >= np.percentile(s2_conf, low)
                    else:
                        mask_low = s2_conf > np.percentile(s2_conf, low)
                    mask_upp = s2_conf <= np.percentile(s2_conf, upp)
                    mask = mask_low & mask_upp
                    inks_conf = np.random.choice(np.array(s2.keys)[mask],
                                                 int(max_sample /
                                                     len(conf_bins)),
                                                 replace=False)
                    inks.extend(inks_conf.tolist())
                inks = sorted(inks)
            # get sign2 and sign3
            _, s2_data = s2.get_vectors(inks)
            _, s3_data = s3.get_vectors(inks)
            _, s3_data_conf = s3.get_vectors(inks, dataset_name='confidence')
            s3_data_conf = s3_data_conf.T[0]
            # get idx of nearest neighbors of s2
            row = dict()
            row['dataset'] = ds
            k = int(s2.shape[0] * 0.0025)
            k = max(k, 10)
            row['k'] = k
            n2_s2 = n2.get_kth_nearest(
                list(s2_data), k=k, distances=False, keys=False)
            n2_s3 = n2.get_kth_nearest(
                list(s3_data), k=k, distances=False, keys=False)

            for low in np.arange(0, 100, 10):
                mask = s3_data_conf >= np.percentile(s3_data_conf, low)
                row['confidence'] = low
                row['jaccard'] = n2.jaccard_similarity(
                    n2_s2['indices'][mask],
                    n2_s3['indices'][mask])
                df = df.append(pd.DataFrame(row), ignore_index=True)

            print(df[df.dataset == ds].groupby('confidence').median())

        sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey='row')
        for ds, ax in zip(self.datasets, axes.flat):
            sns.lineplot(data=df[df.dataset == ds],
                         x='confidence', y='jaccard',
                         style='dataset',
                         markers=False, dashes=False,
                         legend=False,
                         color=self.cc_palette([ds])[0], ax=ax)
            ax.set_xlim(0, 90)
            #ax.set_ylim(0.4, 1)
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.grid(axis='x', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
            ax.set_xlabel('')
            ax.set_ylabel('')
        f.text(0.5, 0.04, 'Confidence', ha='center', va='center')
        f.text(0.06, 0.5, 'Jaccard Neighbors Sign2/3', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_neig2_jaccard.png')
        plt.savefig(outfile, dpi=200)
        plt.close('all')
        return df

    def sign_property_distribution(self, cctype, molset, prop, xlim=None):

        sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey=True)

        for ds, ax in zip(self.datasets, axes.flat):
            try:
                s3 = self.cc.get_signature(cctype, molset, ds)
            except Exception:
                continue
            if not os.path.isfile(s3.data_path):
                continue
            # decide sample molecules
            s3_data_conf = s3.get_h5_dataset(prop)
            # get idx of nearest neighbors of s2
            if xlim:
                sns.distplot(s3_data_conf, color=self.cc_palette([ds])[0],
                             kde=False, norm_hist=False, ax=ax, bins=20,
                             hist_kws={'range': xlim})
                ax.set_xlim(xlim)
            else:
                sns.distplot(s3_data_conf, color=self.cc_palette([ds])[0],
                             kde=False, norm_hist=False, ax=ax)
            ax.set_yscale('log')
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.grid(axis='x', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
        f.text(0.5, 0.04, prop, ha='center', va='center')
        f.text(0.06, 0.5, 'molecules', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, '%s.png' % '_'.join([cctype, molset, prop]))
        plt.savefig(outfile, dpi=200)
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
        sns.set_style("whitegrid")
        sns.set_context("talk")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 15), dpi=100)

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
        plt.savefig(filename, dpi=100)
        plt.close('all')
        return df

    def sign_confidence_distribution(self):

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 10), dpi=100)

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

            df = pd.DataFrame({'train': True, 'confidence': 0[
                              train_idxs], 'kind': 'old'})
            df = df.append(pd.DataFrame({'train': False, 'confidence': s3_conf[
                           test_idxs], 'kind': 'old'}), ignore_index=True)
            df = df.append(pd.DataFrame({'train': True, 'confidence': s3_conf_new[
                           train_idxs], 'kind': 'new'}), ignore_index=True)
            df = df.append(pd.DataFrame({'train': False, 'confidence': s3_conf_new[
                           test_idxs], 'kind': 'new'}), ignore_index=True)
            df = df.append(pd.DataFrame({'train': True, 'confidence': s3_conf_new_w[
                           train_idxs], 'kind': 'test'}), ignore_index=True)
            df = df.append(pd.DataFrame({'train': False, 'confidence': s3_conf_new_w[
                           test_idxs], 'kind': 'test'}), ignore_index=True)
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
            plt.savefig(outfile, dpi=200)
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

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True,
                                 figsize=(10, 10), dpi=100)
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
            plt.savefig(outfile, dpi=200)
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

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(5, 5, sharey=False, sharex=True,
                                 figsize=(10, 10), dpi=100)
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
            plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_test_pearson_distribution(self):

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

        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
        fig, axes = plt.subplots(26, 1, sharex=True, figsize=(3, 10), dpi=100)
        all_dss = list(self.datasets)
        fig.subplots_adjust(left=0.05, right=.95, bottom=0.08,
                            top=1, wspace=0, hspace=-.3)
        colors = [
            '#EA5A49', '#EE7B6D', '#EA5A49', '#EE7B6D', '#EA5A49',
            '#C189B9', '#B16BA8', '#C189B9', '#B16BA8', '#C189B9',
            '#5A72B5', '#7B8EC4', '#5A72B5', '#7B8EC4', '#5A72B5',
            '#96BF55', '#7CAF2A', '#96BF55', '#7CAF2A', '#96BF55',
            '#F39426', '#F5A951', '#F39426', '#F5A951', '#F39426']
        for ds, ax, color in zip(all_dss, axes.flat, colors):
            s3 = self.cc.get_signature('sign3', 'full', ds)
            corr_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_comp.npy')
            if not os.path.isfile(corr_file):
                # filter most correlated spaces
                ds_corr = s3.get_h5_dataset('datasets_correlation')
                #self.__log.info(str(zip(list(self.datasets), list(ds_corr))))
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
                corr_thr = 0.70
                corr_spaces = np.array(list(self.datasets))[
                    ds_corr > corr_thr].tolist()
                #self.__log.info('masking %s' % str(corr_spaces))
                idxs = [all_dss.index(d) for d in corr_spaces]
                x_thr, y_true = mask_exclude(idxs, x_test, y_test)
                y_pred = AdaNet.predict(x_thr, predict_fn)

                corr_test_comp = row_wise_correlation(y_pred.T, y_true.T)
                np.save(corr_file, corr_test_comp)
            corr_test_comp = np.load(corr_file)

            # self.__log.info('%.2f N(%.2f,%.2f)' % (
            #    corr_thr, np.mean(corr_test_comp), np.std(corr_test_comp)))
            #sns.distplot(corr_test_comp, ax=ax,  color=color)
            sns.kdeplot(corr_test_comp, ax=ax, clip_on=False, shade=True,
                        alpha=1, lw=1.5, bw=.2, color=color)
            sns.kdeplot(corr_test_comp, ax=ax, clip_on=False,
                        color="w", lw=2, bw=.2)
            ax.axhline(y=0, lw=2, clip_on=False, color=color)
            ax.set_xlim(0, 1)
            ax.tick_params(axis='x', colors=color)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.patch.set_alpha(0)
            sns.despine(ax=ax, bottom=True, left=True, trim=True)
            '''
            ax.legend(prop={'size': 6})
            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.spines["bottom"].set_color(self.cc_palette([ds])[0])
            ax.spines["top"].set_color(self.cc_palette([ds])[0])
            ax.spines["right"].set_color(self.cc_palette([ds])[0])
            ax.spines["left"].set_color(self.cc_palette([ds])[0])
            '''
            outfile = os.path.join(
                self.plot_path, 'sign3_test_pearson_distribution.png')
        ax = axes.flat[-1]
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
        ax.set_xlabel('Test Correlation',
                      fontdict=dict(name='Arial', size=16))
        ax.tick_params(labelsize=14)
        ax.patch.set_alpha(0)
        sns.despine(ax=ax, bottom=False, left=True, trim=True)

        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_mfp_predictor(self):

        sns.set_style("whitegrid")
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
            sns.barplot(x='component_cat', y='pearson', data=df, hue='split',
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

        f.text(0.06, 0.5, 'Correlation True/Pred. (Pearson)', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_mfp_predictor.png')
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    @staticmethod
    def quick_gaussian_kde(x, y, limit=1000):
        xl = x[:limit]
        yl = y[:limit]
        xy = np.vstack([xl, yl])
        c = gaussian_kde(xy)(xy)
        order = c.argsort()
        return xl, yl, c, order

    def sign3_confidence_summary(self, limit=5000):

        from chemicalchecker.core.signature_data import DataSignature

        sns.set_style("ticks")
        # sns.set_context("paper")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
        f, axes = plt.subplots(5, 5, figsize=(
            10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.08, right=1, bottom=0.08,
                            top=1, wspace=0, hspace=0)
        f.text(0.55, 0.02, 'Error (Log10 MSE)',
               ha='center', va='center',
               name='Arial', size=16)
        f.text(0.02, 0.55, 'Confidence', ha='center',
               va='center', rotation='vertical',
               name='Arial', size=16)
        for ds, ax in zip(self.datasets, axes.flatten()):
            s3 = self.cc.get_signature('sign3', 'full', ds)

            error_dist = DataSignature(os.path.join(s3.model_path, 'error.h5'))
            stddev = error_dist.get_h5_dataset('stddev').flatten()
            intensity = error_dist.get_h5_dataset('intensity').flatten()
            exp_error = error_dist.get_h5_dataset('exp_error').flatten()
            log_mse = error_dist.get_h5_dataset('log_mse')
            log_mse_consensus = error_dist.get_h5_dataset('log_mse_consensus')
            keys = error_dist.get_h5_dataset('keys')

            confidence = s3.get_h5_dataset('confidence_raw').flatten()
            mask = np.isin(list(s3.keys), list(keys), assume_unique=True)
            confidence = confidence[mask]
            pc_confidence = stats.pearsonr(log_mse_consensus, confidence)[0]
            pc_stddev = abs(stats.pearsonr(log_mse_consensus, stddev)[0])
            pc_intensity = abs(stats.pearsonr(log_mse_consensus, intensity)[0])
            pc_exp_error = abs(stats.pearsonr(log_mse, exp_error)[0])
            x, y, c, order = self.quick_gaussian_kde(
                log_mse_consensus, confidence, limit)

            white = self._rgb2hex(250, 250, 250)
            black = self._rgb2hex(0, 0, 0)
            colors = [white, self.cc_palette([ds])[0], black]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                '', colors)
            ax.scatter(x[order], y[order], c=c[order],
                       cmap=cmap, s=5, edgecolor='')
            ax.text(0.05, 0.1, r"$\rho$: {:.2f}".format(pc_confidence),
                    transform=ax.transAxes, name='Arial', size=10,
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.set_ylim(-0.0, 1.1)
            ax.set_xlim(-3.1, -1.4)
            ax.set_ylabel('Confidence')
            ax.set_ylabel('')
            ax.set_xlabel('Error (Log10 MSE)')
            ax.set_xlabel('')

            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(['0', '1'])
            ax.set_xticks([-3.0, -2])
            ax.set_xticklabels(['-3', '-1.5'])
            ax.tick_params(labelsize=14, direction='inout')

            if ds[:2] == 'E1':
                sns.despine(ax=ax, offset=3, trim=True)
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)

            # pies
            colors = [self.cc_palette([ds])[0], 'lightgrey']
            bbox = (.5, .5, .5, .5)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes,  loc=2)

            inset_ax.pie([pc_stddev, 1 - pc_stddev],
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'])
            inset_ax.text(0.5, 0.5, r"$\sigma$", ha='center', va='center',
                          transform=inset_ax.transAxes, name='Arial', size=10)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes,  loc=1)
            inset_ax.pie([pc_intensity, 1 - pc_intensity],
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'])
            inset_ax.text(0.5, 0.5, r"$I$", ha='center', va='center',
                          transform=inset_ax.transAxes, name='Arial', size=10)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes, loc=4)
            inset_ax.pie([pc_exp_error, 1 - pc_exp_error],
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'])
            inset_ax.text(0.5, 0.5, r"$e$", ha='center', va='center',
                          transform=inset_ax.transAxes, name='Arial', size=10)

            outfile = os.path.join(self.plot_path,
                                   'sign3_confidence_summary.png')
            plt.savefig(outfile, dpi=200)
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_novel_confidence_distribution(self):

        sns.set_style("whitegrid")
        f, axes = plt.subplots(5, 5, figsize=(9, 9), sharex=True, sharey=True)

        for ds, ax in zip(self.datasets, axes.flat):
            try:
                s3 = self.cc.get_signature('sign3', 'full', ds)
            except Exception:
                continue
            if not os.path.isfile(s3.data_path):
                continue
            # get novelty and confidence
            #nov = s3.get_h5_dataset('novelty')
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
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_examplary_test_correlation(self, limit=1000,
                                         examplary_ds=['B1.001', 'E2.001', 'D1.001']):

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

        all_dss = list(self.datasets)

        true_pred = dict()
        for ds in examplary_ds:
            s3 = self.cc.get_signature('sign3', 'full', ds)
            corr_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_comp.npy')
            lowcorr_true_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_true.npy')
            lowcorr_pred_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_pred.npy')
            files = [corr_file, lowcorr_true_file, lowcorr_pred_file]
            if not all(os.path.isfile(x) for x in files):
                # filter most correlated spaces
                ds_corr = s3.get_h5_dataset('datasets_correlation')
                #self.__log.info(str(zip(list(self.datasets), list(ds_corr))))
                # load X Y data
                traintest_file = os.path.join(s3.model_path, 'traintest.h5')
                traintest = Traintest(traintest_file, 'test')
                traintest.open()
                x_test, y_test = traintest.get_xy(0, limit)
                traintest.close()
                traintest = Traintest(traintest_file, 'validation')
                traintest.open()
                x_val, y_val = traintest.get_xy(0, limit)
                traintest.close()
                x_test = np.vstack((x_test, x_val))
                y_test = np.vstack((y_test, y_val))
                # load DNN
                predict_fn = AdaNet.predict_fn(os.path.join(
                    s3.model_path, 'adanet_eval', 'savedmodel'))
                # check various correlations thresholds
                corr_thr = 0.70
                corr_spaces = np.array(list(self.datasets))[
                    ds_corr > corr_thr].tolist()
                #self.__log.info('masking %s' % str(corr_spaces))
                idxs = [all_dss.index(d) for d in corr_spaces]
                x_thr, y_true = mask_exclude(idxs, x_test, y_test)
                y_pred = AdaNet.predict(x_thr, predict_fn)

                corr_test_comp = row_wise_correlation(y_pred.T, y_true.T)
                np.save(corr_file, corr_test_comp)
                np.save(lowcorr_true_file, y_true)
                np.save(lowcorr_pred_file, y_pred)
            corr_test_comp = np.load(corr_file)
            y_true = np.load(lowcorr_true_file)
            y_pred = np.load(lowcorr_pred_file)
            self.__log.debug('y_true.shape %s', str(y_true.shape))
            self.__log.debug('y_pred.shape %s', str(y_pred.shape))
            # get median component index
            comp = list(corr_test_comp).index(
                np.percentile(corr_test_comp, 75, interpolation='nearest'))
            y_true_all = y_true[:, comp]
            y_pred_all = y_pred[:, comp]  # .reshape(-1, order='F')
            true_pred[ds] = (y_true_all, y_pred_all,
                             stats.pearsonr(y_true_all, y_pred_all)[0], comp)

        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})
        sns.set_style("ticks")
        fig, axes = plt.subplots(3, 1, figsize=(
            3, 10), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.18, right=1, bottom=0.08,
                            top=1, wspace=0, hspace=0.25)
        fig.text(0.55, 0.02, 'Actual Signature',
                 ha='center', va='center',
                 name='Arial', size=16)
        fig.text(0.04, 0.55, 'Predicted', ha='center',
                 va='center', rotation='vertical',
                 name='Arial', size=16)

        for ds, ax in zip(examplary_ds, axes.flat):

            x, y, c, order = self.quick_gaussian_kde(
                true_pred[ds][0], true_pred[ds][1], limit)

            ax.plot((-1, 1), (-1, 1), ls="--", c=".1", alpha=.6)

            white = self._rgb2hex(250, 250, 250)
            black = self._rgb2hex(0, 0, 0)
            colors = [self.cc_colors(ds, 2), self.cc_palette([ds])[0], black]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                '', colors)
            sns.regplot(x[order], y[order], ax=ax, n_boot=10000,
                        color=self.cc_colors(ds),
                        scatter_kws=dict(cmap=cmap, c=c[order], color=None,
                                         s=10, edgecolor=''))
            ax.text(0.02, 0.9, r"{} comp. {:d}:  $\rho$ {:.2f}".format(
                ds[:2], true_pred[ds][3], true_pred[ds][2]),
                transform=ax.transAxes, name='Arial', size=10,
                bbox=dict(facecolor='white', alpha=0.8))

            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylabel('True')
            ax.set_ylabel('')
            ax.set_xlabel('Pred')
            ax.set_xlabel('')

            ax.set_yticks([-1.0, 0, 1.0])
            ax.set_yticklabels(['-1', '0', '1'])
            ax.set_xticks([-1.0, 0, 1.0])
            ax.set_xticklabels(['-1', '0', '1'])
            ax.tick_params(labelsize=14, direction='inout')
            sns.despine(ax=ax, offset=3, trim=True)

        # plt.tight_layout()
        outfile = os.path.join(
            self.plot_path, 'sign3_examplary_test_correlation.png')
        plt.savefig(outfile, dpi=200)
        plt.close('all')
