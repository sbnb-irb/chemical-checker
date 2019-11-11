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
from matplotlib.collections import LineCollection

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
        stat_file = os.path.join(sign.model_path, 'adanet_eval', 'stats_eval.pkl')
        df = pd.read_pickle(stat_file)
        # merge all stats to pandas
        df = pd.DataFrame(columns=['coordinate'] + list(df.columns))
        for ds in tqdm(self.datasets):
            sign = self.cc.get_signature(ctype, 'full', ds)
            stat_file = os.path.join(sign.model_path, 'adanet_eval', 'stats_eval.pkl')
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

        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02,
                            top=0.95, hspace=0.1)
        gs = fig.add_gridspec(2, 1)
        gs.set_height_ratios((30, 1))
        gs_ds = gs[0].subgridspec(1, 2, wspace=0.1, hspace=0.0)
        ax_cov = fig.add_subplot(gs_ds[0])
        ax_xcov = fig.add_subplot(gs_ds[1])

        cdf = pd.DataFrame([(x[:2], totals[x]) for x in sorted(totals.keys(), reverse=True)],
                           columns=['dataset', 'coverage'])
        ax_cov.barh(range(25), cdf['coverage'],
                    color=list(reversed(colors)), lw=0)
        for y, name in enumerate(cdf['dataset'].tolist()):
            ax_cov.text(2, y - 0.06, name, name='Arial', size=12,
                        va='center', ha='left',
                        color='white', fontweight='bold')
        ax_cov.set_yticks(range(1, 26))
        ax_cov.set_yticklabels([])
        #ax_cov.set_xlim(0, 110)
        # ax_cov.xaxis.set_label_position('top')
        # ax_cov.xaxis.tick_top()
        plt.tick_params(left=False)
        ax_cov.set_ylabel('')
        ax_cov.set_xlabel('Molecules', fontdict=dict(name='Arial', size=16))
        ax_cov.set_xscale('log')
        ax_cov.set_xlim(1, 1e6)
        ax_cov.set_ylim(-1, 25)
        # ax.xaxis.tick_top()
        ax_cov.set_xticks([1e1, 1e3, 1e5])
        ax_cov.set_xticklabels(
            [r'$10^{1}$', r'$10^{3}$',  r'$10^{5}$', ])
        # ax_cov.tick_params(labelsize=14)
        ax_cov.tick_params(left=False, labelsize=14, direction='inout')
        sns.despine(ax=ax_cov, left=True, bottom=True, top=False, trim=True)
        ax_cov.xaxis.set_label_position('top')
        ax_cov.xaxis.tick_top()
        ax_xcov.tick_params(left=False, bottom=False, top=True,
                            labelbottom=False, labeltop=True)
        cmap = plt.get_cmap("Greys_r", 10)
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
                         name='Arial', size=12, ha='center', va='center',
                         color='white', clip_on=False)

        sns.despine(ax=ax_xcov, left=True, bottom=True)
        ax_xcov.tick_params(left=False, bottom=False, top=False,
                            labelbottom=False, labeltop=False)

        ax_cbar = fig.add_subplot(gs[1])
        cbar = matplotlib.colorbar.ColorbarBase(
            ax_cbar, cmap=cmap, orientation='horizontal',
            ticklocation='top')
        cbar.ax.set_xlabel('Overlap',
                           fontdict=dict(name='Arial', size=14))
        cbar.ax.tick_params(labelsize=14, )
        cbar.set_ticks([1, .8, .6, .4, .2, .0])
        cbar.set_ticklabels(
            list(reversed(['1', '0.8', '0.6', '0.4', '0.2', '0'])))

        outfile = os.path.join(self.plot_path, 'sign3_coverage_barplot.png')
        plt.savefig(outfile, dpi=100)
        plt.close('all')

    def sign3_CCA_heatmap(self, limit=10000):

        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        df = pd.DataFrame(columns=['from', 'to', 'CCA'])
        cca_file = os.path.join(self.plot_path, "sign3_CCA.pkl")
        if not os.path.isfile(cca_file):
            for i in range(len(self.datasets)):
                ds_from = self.datasets[i]
                s3_from = self.cc.get_signature(
                    'sign3', 'full', ds_from)[:limit]
                for j in range(i + 1):
                    ds_to = self.datasets[j]
                    s3_to = self.cc.get_signature(
                        'sign3', 'full', ds_to)[:limit]
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
            df.to_pickle(cca_file)
        df = pd.read_pickle(cca_file)
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

        # also plot as MDS projection
        cca = df.pivot('from', 'to', 'CCA').values
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
            ax.annotate('', xy=p1, xycoords='data',
                        xytext=p2, textcoords='data',
                        zorder=1,
                        arrowprops=dict(arrowstyle="-", color="0.2",
                                        shrinkA=10, shrinkB=10,
                                        patchA=None, patchB=None,
                                        connectionstyle="bar,angle=%.2f,fraction=-%.2f" % (
                                            angle, (1 - np.power(coords_dist, 1 / 3))),
                                        ),)

            #ax.plot([p1[0],p2[0]], [p1[1],p2[1]])
            midpoint = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.annotate('%.2f' % dist, xy=midpoint,
                        xytext=midpoint, textcoords='data', xycoords='data',
                        rotation=angle, va='center', ha='center')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        colors = [self.cc_colors(ds) for ds in self.datasets]
        markers = ["$\u2776$", "$\u2777$", "$\u2778$", "$\u2779$", "$\u277a$"]
        for i, color in enumerate(colors):
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=400,
                       edgecolor='', marker=markers[i % 5], zorder=2)
        anno_dist(ax, 19, 15, coords)
        #anno_dist(ax, 20, 22,coords)
        anno_dist(ax, 1, 2, coords)
        anno_dist(ax, 9, 20, coords)
        filename = os.path.join(
            self.plot_path, "sign3_CCA_MDS.svg")
        plt.axis('off')
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

    def sign3_similarity_search(self, limit=1000, compute_confidence=False):

        from chemicalchecker.util.remove_near_duplicates import RNDuplicates
        from chemicalchecker.util.splitter import Traintest
        from chemicalchecker.tool.adanet import AdaNet
        from chemicalchecker.core.sign3 import subsample_x_only
        from scipy.stats import pearsonr
        from sklearn.preprocessing import QuantileTransformer
        from chemicalchecker.core.signature_data import DataSignature


        outfile = os.path.join(
            self.plot_path, 'sign3_simsearch.pkl')
        if not os.path.isfile(outfile):
            df = pd.DataFrame(
                columns=['dataset', 'confidence', 'jaccard', 'k', 'thr'])
            for ds in self.datasets[:]:
                s3 = self.cc.get_signature('sign3', 'full', ds)
                # get traintest split train Y
                traintest_file = os.path.join(s3.model_path, 'traintest.h5')
                traintest = Traintest(traintest_file, 'train')
                traintest.open()
                y_train = traintest.get_y(0, limit)
                x_train = traintest.get_x(0, limit)
                traintest.close()
                print('***y_train', y_train.shape)
                # make train signature fake inks
                s3_train = DataSignature('./tmp/train_%s.h5' % ds)
                # if not os.path.isfile(s3_train.data_path):
                inks = ["{:010d}".format(x) for x in range(len(y_train))]
                with h5py.File(s3_train.data_path, 'w') as hf:
                    hf.create_dataset('V', data=y_train)
                    hf.create_dataset('keys', data=np.array(
                        inks, DataSignature.string_dtype()))
                print('***s3_train', s3_train.info_h5)
                # remove duplicates
                s3_train_ref = DataSignature('./tmp/train_ref_%s.h5' % ds)
                # if not os.path.isfile(s3_train_ref.data_path):
                rnd = RNDuplicates(cpu=4)
                rnd.remove(s3_train.data_path,
                           save_dest=s3_train_ref.data_path)
                print('***s3_train_ref', s3_train_ref.info_h5)
                # make neig
                n2_train = self.cc.get_signature('neig2', 'train', ds)
                # if not os.path.isfile(n2_train.data_path):
                n2_train.fit(s3_train_ref)
                print('***n2_train', n2_train.info_h5)

                pred_file = os.path.join(
                    s3.model_path, 'adanet_eval', 'y_pred_0.90.npy')
                true_file = os.path.join(
                    s3.model_path, 'adanet_eval', 'y_true.npy')
                conf_file = os.path.join(
                    s3.model_path, 'adanet_eval', 'y_pred_0.90_conf.npy')

                y_true = np.load(true_file)[:limit]
                y_pred = np.load(pred_file)[:limit]
                y_rnd = np.random.rand(*y_pred.shape)
                print('***y_true', y_true.shape)
                if compute_confidence and os.path.isfile(conf_file):
                    confidence = np.load(conf_file)[:limit]
                elif compute_confidence:
                    # compute raw confidence scores for train
                    print('Computing raw confidence scores for TRAIN')
                    traintest = Traintest(traintest_file, 'train')
                    traintest.open()
                    x_train = traintest.get_x(0, limit*10)
                    traintest.close()
                    rf = pickle.load(
                        open(os.path.join(s3.model_path,
                                          'adanet_error_final/RandomForest.pkl'), 'rb'), encoding='latin1')
                    predict_fn = s3.get_predict_fn('adanet_eval')
                    nan_feat = np.full((1, x_train.shape[1]), np.nan, dtype=np.float32)
                    nan_pred = predict_fn({'x': nan_feat})['predictions']
                    coverage = ~np.isnan(x_train[:, 0::128])
                    err_dist = rf.predict(coverage)
                    pred, samples = AdaNet.predict(x_train, predict_fn,
                                                   subsample_x_only,
                                                   consensus=True,
                                                   samples=10)
                    consensus = np.mean(samples, axis=1)
                    centered = consensus - nan_pred
                    intensities = np.abs(centered)
                    int_dist = np.mean(intensities, axis=1).flatten()
                    stddevs = np.std(samples, axis=1)
                    std_dist = np.mean(stddevs, axis=1).flatten()
                    log_mse = np.log10(np.mean(((y_train - pred)**2), axis=1))
                    log_mse_consensus = np.log10(
                        np.mean(((y_train - consensus)**2), axis=1))
                    # get calibration weights
                    print('calibrationg weights')
                    corr_int = pearsonr(int_dist.flatten(), log_mse_consensus)[0]
                    corr_std = pearsonr(std_dist.flatten(), log_mse_consensus)[0]
                    corr_err = pearsonr(err_dist.flatten(), log_mse)[0]
                    # normalizers
                    print('normalizers')
                    std_qtr = QuantileTransformer(
                        n_quantiles=len(std_dist)).fit(np.expand_dims(std_dist, 1))
                    int_qtr = QuantileTransformer(
                        n_quantiles=len(int_dist)).fit(np.expand_dims(int_dist, 1))
                    err_qtr = QuantileTransformer(
                        n_quantiles=len(err_dist)).fit(np.expand_dims(err_dist, 1))

                    # get raw confidence scores for test
                    print('Computing raw confidence scores for TEST')
                    traintest = Traintest(traintest_file, 'test')
                    traintest.open()
                    x_test = traintest.get_x(0, limit)
                    traintest.close()
                    coverage = ~np.isnan(x_test[:, 0::128])
                    errors = rf.predict(coverage)
                    error_norm = err_qtr.transform(np.expand_dims(errors, 1))
                    pred, samples = AdaNet.predict(x_test, predict_fn,
                                                   subsample_x_only,
                                                   consensus=True,
                                                   samples=10)
                    consensus = np.mean(samples, axis=1)
                    centered = consensus - nan_pred
                    intensities = np.abs(centered)
                    intensities = np.mean(intensities, axis=1)
                    inten_norm = int_qtr.transform(np.expand_dims(intensities, 1))
                    stddevs = np.std(samples, axis=1)
                    stddevs = np.mean(stddevs, axis=1)
                    stddev_norm = std_qtr.transform(np.expand_dims(stddevs, 1))

                    confidence = np.average(
                        [inten_norm, (1 - stddev_norm), (1 - error_norm)],
                        weights=[corr_int, corr_std, corr_err], axis=0)
                    confidence = confidence.flatten()
                    np.save(conf_file, confidence)
                    print('Confidence DONE')
                else:
                    confidence = np.mean(abs(y_pred), axis=1)
                rnd_confidence = np.random.rand(len(confidence))

                for k in [1, 5, 10, 50, 100]:
                    # get idxs of nearest neighbors of s2 and s3
                    n2_s2 = n2_train.get_kth_nearest(
                        list(y_true), k=k, distances=True, keys=True)
                    n2_s3 = n2_train.get_kth_nearest(
                        list(y_pred), k=k, distances=True, keys=False)
                    # get jaccard
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'], n2_s3['indices'])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'all',
                         'jaccard': jacc, 'k': k, 'thr': 1}), ignore_index=True)
                    print('*****  ALL', k, len(jacc), np.mean(jacc))
                    # only consider jaccard of confident predictions
                    topn = int(np.floor(len(y_true) / 10))
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'][np.argsort(confidence)[-topn:]],
                        n2_s3['indices'][np.argsort(confidence)[-topn:]])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'high',
                         'jaccard': jacc, 'k': k, 'thr': 1}), ignore_index=True)
                    # random confidence
                    topn = int(np.floor(len(y_true) / 10))
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'][np.argsort(rnd_confidence)[-topn:]],
                        n2_s3['indices'][np.argsort(rnd_confidence)[-topn:]])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'rnd_conf',
                         'jaccard': jacc, 'k': k, 'thr': 1}), ignore_index=True)
                    # random signatures
                    n2_rnd = n2_train.get_kth_nearest(
                        list(y_rnd), k=k, distances=True, keys=False)
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'], n2_rnd['indices'])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'rnd_sign',
                         'jaccard': jacc, 'k': k, 'thr': 1}), ignore_index=True)

                for thr in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
                    k = 10
                    # get idxs of nearest neighbors of s2 and s3
                    n2_s2 = n2_train.get_kth_nearest(
                        list(y_true), k=10, distances=True, keys=True)
                    n2_s3 = n2_train.get_kth_nearest(
                        list(y_pred), k=10, distances=True, keys=False)
                    # drop searches that are above threshold
                    mask = np.any(n2_s2['distances'] < thr, axis=1)
                    for key, val in n2_s2.items():
                        n2_s2[key] = val[mask]
                    for key, val in n2_s3.items():
                        n2_s3[key] = val[mask]
                    # only consider neighbors below threshold
                    thr_n2_s2 = [n2_s2['indices'][x][n2_s2['distances'][x] < thr]
                                 for x in range(n2_s2['indices'].shape[0])]
                    thr_n2_s3 = [n2_s3['indices'][x][n2_s3['distances'][x] < thr]
                                 for x in range(n2_s3['indices'].shape[0])]
                    # get jaccard
                    jacc = n2_train.jaccard_similarity(
                        thr_n2_s2, thr_n2_s3)
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'all',
                         'jaccard': jacc, 'k': k, 'thr': thr}), ignore_index=True)
                    print('*****  ALL', k, len(jacc), np.mean(jacc))
                    # only consider jaccard of confident predictions
                    topn = int(np.floor(np.count_nonzero(mask) / 10))
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'][np.argsort(confidence[mask])[-topn:]],
                        n2_s3['indices'][np.argsort(confidence[mask])[-topn:]])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'high',
                         'jaccard': jacc, 'k': k, 'thr': thr}), ignore_index=True)
                    # random confidence
                    topn = int(np.floor(np.count_nonzero(mask) / 10))
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'][np.argsort(rnd_confidence[mask])[-topn:]],
                        n2_s3['indices'][np.argsort(rnd_confidence[mask])[-topn:]])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'rnd_conf',
                         'jaccard': jacc, 'k': k, 'thr': thr}), ignore_index=True)
                    # random signatures
                    n2_rnd = n2_train.get_kth_nearest(
                        list(y_rnd), k=k, distances=True, keys=False)
                    thr_n2_rnd = [n2_rnd['indices'][x][n2_rnd['distances'][x] < thr]
                                  for x in range(n2_rnd['indices'].shape[0])]
                    jacc = n2_train.jaccard_similarity(
                        n2_s2['indices'], n2_rnd['indices'])
                    df = df.append(pd.DataFrame(
                        {'dataset': ds, 'confidence': 'rnd_sign',
                         'jaccard': jacc, 'k': k, 'thr': thr}), ignore_index=True)

            df.to_pickle(outfile)
        df = pd.read_pickle(outfile)
        df['k'] = df.k.astype('category')
        df['thr'] = df.thr.astype('category')

        # quick check on coherence
        for ds in self.datasets:
            a = df[(df.dataset == ds) & (df.k == 10)
                   & (df.thr == 1) & (df.confidence != 'rnd_sign')
                   & (df.confidence != 'rnd_conf')].jaccard.values
            x = int(len(a) / 2)
            assert(np.all(a[-x:] == a[x:]))

        sns.set_style("ticks")
        f, axes = plt.subplots(5, 5, figsize=(5, 6), sharex=True, sharey='row')
        plt.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.99,
                            wspace=.08, hspace=.1)
        for ds, ax in zip(self.datasets[:], axes.flat):
            sns.pointplot(data=df[df.dataset == ds], y='jaccard', x='k',
                          hue_order=['rnd_sign', 'rnd_conf', 'all', 'high'], hue='confidence',
                          linestyles=['--', '--', '-', '-'],
                          markers=[',', '', 'o', 'o'],
                          ax=ax, scale=0.4, errwidth=0.5, dodge=0.3,
                          palette=['grey', 'grey', self.cc_colors(ds, 2), self.cc_colors(ds, 0)])
            ax.get_legend().remove()

            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(0, 1)
            if ds[:2] == 'E1':
                sns.despine(ax=ax, offset=3, trim=True)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
                ax.tick_params(axis='x', labelrotation=45)
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
                ax.tick_params(axis='x', labelrotation=45)
                #ax.set_xticks([0, 1])
                #ax.set_xticklabels(['All', 'High'])
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)
        f.text(0.5, 0.04, 'Neighbors Searched', ha='center', va='center')
        f.text(0.06, 0.5, 'Jaccard Similarity', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_simsearch_neig.png')
        plt.savefig(outfile, dpi=200)
        plt.close('all')

        sns.set_style("ticks")
        f, axes = plt.subplots(5, 5, figsize=(5, 6), sharex=True, sharey='row')
        plt.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.99,
                            wspace=.08, hspace=.1)
        for ds, ax in zip(self.datasets[:], axes.flat):
            sns.pointplot(data=df[df.dataset == ds], y='jaccard', x='thr',
                          hue_order=['rnd_sign', 'rnd_conf', 'all', 'high'], hue='confidence',
                          linestyles=['--', '--', '-', '-'],
                          markers=[',', '', 'o', 'o'],
                          ax=ax, scale=0.4, errwidth=0.5, dodge=0.3,
                          palette=['grey', 'grey', self.cc_colors(ds, 2), self.cc_colors(ds, 0)])
            ax.get_legend().remove()

            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(0, 1)
            if ds[:2] == 'E1':
                sns.despine(ax=ax, offset=3, trim=True)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
                ax.tick_params(axis='x', labelrotation=45)
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['0', '1'])
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
                ax.tick_params(axis='x', labelrotation=45)
                #ax.set_xticks([0, 1])
                #ax.set_xticklabels(['All', 'High'])
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)
        f.text(0.5, 0.04, 'Distance Threshold', ha='center', va='center')
        f.text(0.06, 0.5, 'Jaccard Similarity', ha='center',
               va='center', rotation='vertical')
        outfile = os.path.join(
            self.plot_path, 'sign3_simsearch_thr.png')
        plt.savefig(outfile, dpi=200)
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
            df = df.append(pd.DataFrame(
                {'dataset': ds, 'confidence': 'high', 'jaccard': jacc}), ignore_index=True)
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
            df = df.append(pd.DataFrame(
                {'dataset': ds, 'confidence': 'all', 'jaccard': jacc}), ignore_index=True)
            print('***** ALL', len(jacc), np.mean(jacc),
                  stats.spearmanr(jacc, s3_data_conf))

        sns.set_style("whitegrid")
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
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign_property_distribution(self, cctype, molset, prop, xlim=None):
        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        fig, axes = plt.subplots(5, 5, figsize=(5, 5),
                                 sharex=True, sharey=True)

        plt.subplots_adjust(left=0.12, right=0.99, bottom=0.1, top=0.99,
                            wspace=.08, hspace=.06)

        fig.text(0.5, 0.02, prop.capitalize(),
                 ha='center', va='center',
                 name='Arial', size=16)
        fig.text(0.02, 0.55, 'Molecules', ha='center',
                 va='center', rotation='vertical',
                 name='Arial', size=16)

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
                             kde=False, norm_hist=False, ax=ax, bins=10,
                             hist_kws=dict(range=xlim, alpha=1))
                ax.set_xlim(xlim)
            else:
                sns.distplot(s3_data_conf, color=self.cc_palette([ds])[0],
                             kde=False, norm_hist=False, ax=ax)
            ax.set_yscale('log')

            ax.grid(axis='y', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            ax.grid(axis='x', linestyle="-",
                    color=self.cc_palette([ds])[0], lw=0.3)
            if ds[:2] == 'E1':
                sns.despine(ax=ax, offset=3, trim=True)
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(['0', '0.5', '1'])
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)

        plt.minorticks_off()

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

    def sign3_test_distribution(self, limit=10000):

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

        df = pd.DataFrame(columns=['dataset', 'scaled', 'comp_wise',
                                   'metric', 'value', 'corr_thr'])
        all_dss = list(self.datasets)
        for ds in all_dss:
            s3 = self.cc.get_signature('sign3', 'full', ds)
            pred_file = os.path.join(
                s3.model_path, 'adanet_eval', 'y_pred_%.2f.npy')
            true_file = os.path.join(
                s3.model_path, 'adanet_eval', 'y_true.npy')
            if not all([os.path.isfile(pred_file % t) for t in [0.7, 0.9]]):
                # filter most correlated spaces
                ds_corr = s3.get_h5_dataset('datasets_correlation')
                # self.__log.info(str(zip(list(self.datasets), list(ds_corr))))
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
                for corr_thr in [0.7, 0.9]:
                    corr_spaces = np.array(list(self.datasets))[
                        ds_corr > corr_thr].tolist()
                    # self.__log.info('masking %s' % str(corr_spaces))
                    idxs = [all_dss.index(d) for d in corr_spaces]
                    x_thr, y_true = mask_exclude(idxs, x_test, y_test)
                    y_pred = AdaNet.predict(x_thr, predict_fn)

                    np.save(pred_file % corr_thr, y_pred)
                    np.save(true_file, y_true)
            options = [
                ['log10MSE', 'R2', 'Pearson'],
                [0.7, 0.9],
                [True, False],
                [True, False]
            ]
            all_dfs = list()
            for metric, corr_thr, scaled, comp_wise in itertools.product(*options):
                y_true = np.load(true_file)
                y_pred = np.load(pred_file % corr_thr)
                if scaled:
                    scaler = RobustScaler()
                    y_true = scaler.fit_transform(y_true)
                    y_pred = scaler.fit_transform(y_pred)
                if comp_wise:
                    y_true = y_true.T
                    y_pred = y_pred.T
                if metric == 'log10MSE':
                    values = np.log10(np.mean((y_true - y_pred)**2, axis=1))
                elif metric == 'R2':
                    values = r2_score(y_true, y_pred, multioutput='raw_values')
                elif metric == 'Pearson':
                    values = row_wise_correlation(y_true, y_pred)
                _df = pd.DataFrame(dict(dataset=ds, scaled=scaled, comp_wise=comp_wise,
                                        metric=metric, corr_thr=corr_thr, value=values))
                all_dfs.append(_df)
            df = pd.concat(all_dfs)

        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif', 'font.serif': ['Arial']})

        options = [
            ['log10MSE', 'R2', 'Pearson'],
            [0.9],
            [True, False],
            [True, False]
        ]
        for metric, corr_thr, scaled, comp_wise in itertools.product(*options):
            odf = df[(df.scaled == scaled) & (df.comp_wise == comp_wise) & (
                df.metric == metric) & (df.corr_thr == corr_thr)]
            xmin = np.floor(np.percentile(odf.value, 5))
            xmax = np.ceil(np.percentile(odf.value, 95))
            fig, axes = plt.subplots(
                26, 1, sharex=True, figsize=(3, 10), dpi=100)
            fig.subplots_adjust(left=0.05, right=.95, bottom=0.08,
                                top=1, wspace=0, hspace=-.3)
            for idx, (ds, ax) in enumerate(zip(all_dss, axes.flat)):
                color = self.cc_colors(ds, idx % 2)
                color2 = self.cc_colors(ds, 2)
                values = odf[(odf.dataset == ds)].value.tolist()
                sns.kdeplot(values, ax=ax, clip_on=False, shade=True,
                            alpha=1, lw=0,  bw=.15, color=color)
                sns.kdeplot(values, ax=ax, clip_on=False,
                            color=color2, lw=2, bw=.15)

                ax.axhline(y=0, lw=2, clip_on=False, color=color)
                ax.set_xlim(xmin, xmax)
                ax.tick_params(axis='x', colors=color)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.patch.set_alpha(0)
                sns.despine(ax=ax, bottom=True, left=True, trim=True)
            ax = axes.flat[-1]
            ax.set_yticks([])
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(np.linspace(xmin, xmax, 3))
            ax.set_xticklabels(
                ['%.1f' % x for x in np.linspace(xmin, xmax, 3)])
            # ax.set_xticklabels(['0','0.5','1'])

            xlabel = '%.1f' % corr_thr
            xlabel += ' %s' % metric
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
            fname = 'sign3_test_dist_%s_%.1f' % (metric, corr_thr)
            if comp_wise:
                fname += '_comp'
            if scaled:
                fname += '_scaled'
            print(fname)
            print(odf.value.describe())
            outfile = os.path.join(self.plot_path, fname + '.png')
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
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_mfp_confidence_predictor(self):

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
            colors = [white, self.cc_palette([ds])[0]]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                '', colors)
            ax.scatter(x[order], y[order], c=c[order],
                       cmap=cmap, s=5, edgecolor='', alpha=.8)
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
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[1] == '1':
                sns.despine(ax=ax, bottom=True, offset=3, trim=True)
                ax.tick_params(bottom=False)
                # set the alignment for outer ticklabels
                ticklabels = ax.get_yticklabels()
                ticklabels[0].set_va("bottom")
                ticklabels[-1].set_va("top")
            elif ds[0] == 'E':
                sns.despine(ax=ax, left=True, offset=3, trim=True)
                ax.tick_params(left=False)
            else:
                sns.despine(ax=ax, bottom=True, left=True, offset=3, trim=True)
                ax.tick_params(bottom=False, left=False)

            # pies
            wp = {'linewidth': 0, 'antialiased': True}
            colors = [self.cc_palette([ds])[0], 'lightgrey']
            bbox = (.5, .5, .5, .5)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes,  loc=2)

            inset_ax.pie([pc_stddev, 1 - pc_stddev], wedgeprops=wp,
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'], wedgeprops=wp)
            inset_ax.text(0.5, 0.5, r"$\sigma$", ha='center', va='center',
                          transform=inset_ax.transAxes, name='Arial', size=10)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes,  loc=1)
            inset_ax.pie([pc_intensity, 1 - pc_intensity], wedgeprops=wp,
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'], wedgeprops=wp)
            inset_ax.text(0.5, 0.5, r"$I$", ha='center', va='center',
                          transform=inset_ax.transAxes, name='Arial', size=10)
            inset_ax = inset_axes(ax, 0.4, 0.4,
                                  bbox_to_anchor=bbox,
                                  bbox_transform=ax.transAxes, loc=4)
            inset_ax.pie([pc_exp_error, 1 - pc_exp_error], wedgeprops=wp,
                         counterclock=False, startangle=90, colors=colors)
            inset_ax.pie([1.0], radius=0.5, colors=['white'], wedgeprops=wp)
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
        plt.savefig(outfile, dpi=200)
        plt.close('all')

    def sign3_examplary_test_correlation(self, limit=1000,
                                         examplary_ds=['B1.001', 'D1.001', 'E4.001']):

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

        def plot_molecule(ax, inchikey, size=(300, 300), margin=5, interpolation=None):
            from chemicalchecker.util.parser import Converter
            from rdkit.Chem import Draw
            from rdkit import Chem
            from PIL import Image

            # get smiles
            converter = Converter()
            smiles = converter.inchi_to_smiles(
                eval(converter.inchikey_to_inchi(inchikey))[0]['standardinchi'])
            # generate initial image
            image = Draw.MolToImage(Chem.MolFromSmiles(smiles), size=size)
            # find crop box
            image_data = np.asarray(image)
            image_white = np.all((image_data == [255, 255, 255]), axis=2)
            non_white_columns = np.where(np.sum(~image_white, axis=0) != 0)[0]
            non_white_rows = np.where(np.sum(~image_white, axis=1) != 0)[0]
            cropBox = (min(non_white_rows) - margin,
                       max(non_white_rows) + margin,
                       min(non_white_columns) - margin,
                       max(non_white_columns) + margin)
            image_data_new = image_data[cropBox[0]:cropBox[1] + 1,
                                        cropBox[2]:cropBox[3] + 1, :]
            new_image = Image.fromarray(image_data_new)
            ax.imshow(new_image, interpolation=interpolation)
            ax.set_axis_off()

        all_dss = list(self.datasets)

        true_pred = dict()
        for ds in examplary_ds:
            s2 = self.cc.get_signature('sign2', 'full', ds)
            s3 = self.cc.get_signature('sign3', 'full', ds)
            lowcorr_true_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_true.npy')
            lowcorr_pred_file = os.path.join(
                s3.model_path, 'adanet_eval', 'corr_test_pred.npy')
            files = [lowcorr_true_file, lowcorr_pred_file]
            if not all(os.path.isfile(x) for x in files):
                # filter most correlated spaces
                ds_corr = s3.get_h5_dataset('datasets_correlation')
                # self.__log.info(str(zip(list(self.datasets), list(ds_corr))))
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
                # self.__log.info('masking %s' % str(corr_spaces))
                idxs = [all_dss.index(d) for d in corr_spaces]
                x_thr, y_true = mask_exclude(idxs, x_test, y_test)
                y_pred = AdaNet.predict(x_thr, predict_fn)

                np.save(lowcorr_true_file, y_true)
                np.save(lowcorr_pred_file, y_pred)
            y_true = np.load(lowcorr_true_file)
            y_pred = np.load(lowcorr_pred_file)
            inks = list()
            comp1 = s2[:][:, 0]
            for i in range(y_true.shape[0]):
                ink_i = s2.keys[np.argwhere(comp1 == y_true[i, 0])].flatten()
                inks.append(ink_i.tolist())
            #self.__log.debug('y_true.shape %s', str(y_true.shape))
            #self.__log.debug('y_pred.shape %s', str(y_pred.shape))
            ink_set = set([item for sublist in inks for item in sublist])
            true_pred[ds] = (inks, ink_set,
                             row_wise_correlation(y_true, y_pred),
                             y_true, y_pred)

        shared_inks = set.intersection(*[x[1] for x in true_pred.values()])
        mol_corr = dict()
        for mol in shared_inks:
            mol_corr[mol] = dict()
            for ds, (inks, ink_set, corr, y_true, y_pred) in true_pred.items():
                idx = np.argwhere([mol in x for x in inks])[0][0]
                mol_corr[mol][ds] = (idx, corr[idx], y_true[idx], y_pred[idx])

        for mol in tqdm(mol_corr):  # = 'MBUVEWMHONZEQD-UHFFFAOYSA-N'
            sns.set_style("ticks")
            sns.set_style({'font.family': 'sans-serif',
                           'font.serif': ['Arial']})
            fig = plt.figure(figsize=(3, 10))
            plt.subplots_adjust(left=0.2, right=1, bottom=0.08, top=1)
            gs = fig.add_gridspec(4, 1)
            gs.set_height_ratios((1, 2, 2, 2))
            fig.text(0.5, 0.02, 'Actual Signature',
                     ha='center', va='center',
                     name='Arial', size=16)
            fig.text(0.04, 0.45, 'Predicted', ha='center',
                     va='center', rotation='vertical',
                     name='Arial', size=16)
            ax_mol = fig.add_subplot(gs[0])
            plot_molecule(ax_mol, mol, size=(6000, 6000),
                          interpolation='hanning')
            gss_ds = [gs[1], gs[2], gs[3]]
            for ds, sub in zip(examplary_ds, gss_ds):
                gs_ds = sub.subgridspec(2, 2, wspace=0.0, hspace=0.0)
                gs_ds.set_height_ratios((1, 5))
                gs_ds.set_width_ratios((5, 1))
                ax_main = fig.add_subplot(gs_ds[1, 0])
                ax_top = fig.add_subplot(gs_ds[0, 0], sharex=ax_main)
                ax_top.text(0.05, 0.2, "%s" % ds[:2],
                            color=self.cc_colors(ds),
                            transform=ax_top.transAxes,
                            name='Arial', size=14, weight='bold')
                ax_top.set_axis_off()
                ax_right = fig.add_subplot(gs_ds[1, 1], sharey=ax_main)
                ax_right.set_axis_off()

                ax_main.plot((-1, 1), (-1, 1), ls="--",
                             c="lightgray", alpha=.5)

                true = mol_corr[mol][ds][2]
                pred = mol_corr[mol][ds][3]
                sns.regplot(true, pred,
                            ax=ax_main, n_boot=10000, truncate=False,
                            color=self.cc_colors(ds),
                            scatter_kws=dict(s=10, edgecolor=''),
                            line_kws=dict(lw=1))

                error = np.log10(np.mean((true - pred)**2))
                ax_main.text(0.5, 0.04, r"Error:" + " {:.2f}".format(error),
                             transform=ax_main.transAxes,
                             name='Arial', size=10,
                             bbox=dict(facecolor='white', alpha=0.8))

                sns.despine(ax=ax_main, offset=3, trim=True)

                ax_main.set_ylabel('')
                ax_main.set_xlabel('')
                ax_main.set_yticks([-1.0, 0, 1.0])
                ax_main.set_yticklabels(['-1', '0', '1'])
                ax_main.set_xticks([-1.0, 0, 1.0])
                ax_main.set_xticklabels(['-1', '0', '1'])
                ax_main.tick_params(labelsize=14, direction='inout')

                sns.distplot(true, ax=ax_top,
                             hist=False, kde_kws=dict(shade=True, bw=.2),
                             color=self.cc_colors(ds))
                sns.distplot(pred, ax=ax_right, vertical=True,
                             hist=False, kde_kws=dict(shade=True, bw=.2),
                             color=self.cc_colors(ds))

            # plt.tight_layout()
            spaces = '-'.join([ds[:2] for ds in examplary_ds])
            outfile = os.path.join(
                self.plot_path, 'sign3_%s_%s.png' % (spaces, mol))
            plt.savefig(outfile, dpi=200)
            plt.close('all')

    def sign3_sign2_comparison(self):

        pklfile = os.path.join(self.plot_path, 'sign3_sign2_comparison.pkl')
        if not os.path.isfile(pklfile):
            data = dict()
            for ds in self.datasets:
                if 'dataset' not in data:
                    data['dataset'] = list()
                data['dataset'].append(ds[:2])
                # sign2
                s2 = self.cc.get_signature('sign2', 'full', ds)
                stat_file = os.path.join(
                    s2.stats_path, 'validation_stats.json')
                if not os.path.isfile(stat_file):
                    s2.validate()
                stats = json.load(open(stat_file, 'r'))
                for k, v in stats.items():
                    if k + '_sign2' not in data:
                        data[k + '_sign2'] = list()
                    data[k + '_sign2'].append(v)
                # sign3
                s3 = self.cc.get_signature('sign3', 'full', ds)
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
                        'sign3', 'conf%.1f' % thr, ds)
                    stat_file = os.path.join(s3_conf.stats_path,
                                             'validation_stats.json')
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
        sns.set_style("ticks")
        sns.set_style({'font.family': 'sans-serif',
                       'font.serif': ['Arial']})

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
            start = df[df.dataset == ds[:2]]['moa_cov_sign2'].tolist()[0]
            end = df[df.dataset == ds[:2]]['moa_cov_0.0'].tolist()[0]
            js = ['sign2'] + ['%.1f' %
                              f for f in reversed(np.arange(0.0, 0.9, 0.1))]
            #js = ['sign2','0.5','0.0']
            covs = [df[df.dataset == ds[:2]]['moa_cov_%s' % j].tolist()[0]
                    for j in js]
            cmap = plt.get_cmap("plasma", len(covs))
            gradient_arrow(ax_cov, (start, y), (end, y), xs=covs, lw=7)

            start = df[df.dataset == ds[:2]]['moa_auc_sign2'].tolist()[0]
            aucs = [df[df.dataset == ds[:2]]['moa_auc_%s' % j].tolist()[0]
                    for j in js]
            end = aucs[np.argmax(covs)]
            cmap = plt.get_cmap("plasma", len(covs))
            ax_roc.scatter(start, y, color=cmap(0), s=60)
            ax_roc.scatter(end, y, color=cmap(np.argmax(covs)), s=60)

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
            self.plot_path, 'sign3_sign2_moa_comparison.png')
        plt.savefig(outfile, dpi=100)
        plt.close('all')
