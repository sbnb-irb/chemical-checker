"""Utility for plotting Chemical Checker data."""

import os
import json
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from chemicalchecker.util import logged


matplotlib.use('Agg')


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

    def sign2_adanet_stats(self, metric=None, compare=None):
        # read stats fields
        sign2 = self.cc.get_signature('sign2', 'reference', 'E5.001')
        stat_file = os.path.join(sign2.model_path, 'adanet', 'stats.pkl')
        df = pd.read_pickle(stat_file)
        # merge all stats to pandas
        df = pd.DataFrame(columns=['coordinate'] + list(df.columns))
        for ds in tqdm(self.datasets):
            sign2 = self.cc.get_signature('sign2', 'reference', ds)
            stat_file = os.path.join(sign2.model_path, 'adanet', 'stats.pkl')
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
                    ax.set_ylim(0, 0.2)
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
