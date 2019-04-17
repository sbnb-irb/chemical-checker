"""Utility for plotting Chemical Checker data."""

import os
import h5py
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt

from chemicalchecker.util import logged


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
            "MCC"]
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
        g.axes.flat[0].set_xlim(1e2, 1e7)
        g.axes.flat[0].set_xlabel("Nodes")
        g.axes.flat[1].set_xscale("log")
        g.axes.flat[1].set_xlim(1e3, 1e8)
        g.axes.flat[1].set_xlabel("Edges")
        g.axes.flat[3].set_xlim(min(df["Connected Components"].dropna()), 1.0)
        g.axes.flat[-4].set_xscale("log")
        g.axes.flat[-4].set_xlim(1)
        # g.axes.flat[-1].set_xlim(1e1,1e3)
        sns.despine(left=True, bottom=True)

        outfile = os.path.join(self.plot_path, 'sign2_node2vec_stats.png')
        plt.savefig(outfile, dpi=100)
        plt.close('all')

    def sign2_feature_distribution_plot(self, sample_size=10000):
        fig, axes = plt.subplots(25, 1, sharey=True, sharex=True,
                                 figsize=(10, 40), dpi=100)
        for ds, ax in tqdm(zip(self.datasets, axes.flatten())):
            sign2 = self.cc.get_signature('sign2', 'reference', ds)
            if sign2.shape[0] > sample_size:
                keys = np.random.choice(sign2.keys, sample_size, replace=False)
                matrix = sign2.get_vectors(keys)[1]
            else:
                matrix = sign2[:]
            df = pd.DataFrame(matrix).melt()
            sns.pointplot(x='variable', y='value', data=df,
                          ax=ax, ci='sd', join=False, markers='.',
                          color=self.cc_palette([ds])[0])
            ax.set_ylim(-1, 1)
            ax.set_xlim(-2, 130)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel(ds)
            min_mean = min(np.mean(matrix, axis=0))
            max_mean = max(np.mean(matrix, axis=0))
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
        filename = os.path.join(self.plot_path, "feat_distrib.png")
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
        #odf['capped_train_size'] = np.minimum(odf.train_size,20000)
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

    def spy_matrix(self, matrix):
        present = (~np.isnan(matrix)).astype(int)
        fig, ax = plt.subplots()
        ax.spy(present)
        ax.set_xticks(np.arange(0, 3200, 128))
        ax.set_xticklabels([ds[:2] for ds in list(self.cc.datasets)])
        filename = os.path.join(self.plot_path, "spy.png")
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
