"""Plot comparison of different Chemical Checker versions."""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from chemicalchecker.util import logged


@logged
class MultiCCPlot():
    """MultiCCPlot class.

    Produce Chemical Checker plots using multiple datasets.
    """

    def __init__(self, chemcheckers, names, plot_path, limit_dataset=None):
        """Initialize a MultiCCPlot instance.

        Args:
            chemcheckers (list): List of CC instances.
            names (list): List of CC instances names.
            plot_path (str): Destination folder for plot images.
            limit_dataset (list): Limit plot to these datasets. If None
                all datasets are used.
        """
        if not os.path.isdir(plot_path):
            raise Exception("Folder to save plots does not exist")
        self.__log.debug('Plots will be saved to %s', plot_path)
        self.plot_path = plot_path
        self.ccs = chemcheckers
        self.names = names
        if not limit_dataset:
            self.datasets = list(self.ccs[0].datasets)
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
                           colors_rgba[i, ki]) for i in range(N + 1)]
        # Return colormap object.
        return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

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
        df = pd.DataFrame(columns=['dataset', "CC"] + stats)
        for name, cc in zip(self.names, self.ccs):
            for ds in tqdm(self.datasets):
                # get sign2 and stats file
                sign2 = cc.get_signature('sign2', 'reference', ds)
                graph_file = os.path.join(sign2.stats_path, "graph_stats.json")
                if not os.path.isfile(graph_file):
                    self.__log.warn('Graph stats %s not found', graph_file)
                    df.loc[len(df)] = pd.Series({"dataset": ds, "CC": name})
                    continue
                graph_stat = json.load(open(graph_file, 'r'))
                linkpred_file = os.path.join(sign2.stats_path, "linkpred.json")
                skip_linkpred = False
                if not os.path.isfile(linkpred_file):
                    self.__log.warn(
                        'Node2vec stats %s not found', linkpred_file)
                    skip_linkpred = True
                    pass
                if not skip_linkpred:
                    liknpred_perf = json.load(open(linkpred_file, 'r'))
                    liknpred_perf = {k: float(v)
                                     for k, v in liknpred_perf.items()}
                # prepare row
                for deg in degrees:
                    row = dict()
                    row.update(graph_stat)
                    if not skip_linkpred:
                        row.update(liknpred_perf)
                    row.update({"dataset": ds})
                    row.update({"CC": name})
                    row.update({"Degree": graph_stat[deg]})
                    df.loc[len(df)] = pd.Series(row)
                for conn in conncompo:
                    row = dict()
                    row.update(graph_stat)
                    if not skip_linkpred:
                        row.update(liknpred_perf)
                    row.update({"dataset": ds})
                    row.update({"CC": name})
                    row.update({"Connected Components": graph_stat[conn]})
                    df.loc[len(df)] = pd.Series(row)
                for wei in weights:
                    row = dict()
                    row.update(graph_stat)
                    if not skip_linkpred:
                        row.update(liknpred_perf)
                    row.update({"dataset": ds})
                    row.update({"CC": name})
                    row.update({"Weights": graph_stat[wei]})
                    df.loc[len(df)] = pd.Series(row)
                try:
                    maxss = list()
                    minss = list()
                    for s in sign2.chunker(size=100000):
                        curr = sign2[s]
                        maxss.append(np.percentile(curr, 99))
                        minss.append(np.percentile(curr, 1))
                    row = {"dataset": ds, "CC": name,
                           "Sign Range": np.mean(maxss)}
                    df.loc[len(df)] = pd.Series(row)
                    row = {"dataset": ds, "CC": name,
                           "Sign Range": np.mean(minss)}
                    df.loc[len(df)] = pd.Series(row)
                except:
                    self.__log.warn("SKIPPING %s range" % ds)
                    continue

        df = df.infer_objects()
        sns.set(style="ticks")
        sns.set_context("talk", font_scale=1.)
        g = sns.PairGrid(df.sort_values("dataset", ascending=True),
                         x_vars=stats, y_vars=["dataset"], hue="CC",
                         hue_kws={"marker": [6, "x"]},
                         height=10, aspect=.3)
        g.map(sns.stripplot, size=10, dodge=False, jitter=False,  # marker="|",
              palette=self.cc_palette(self.datasets),
              orient="h", linewidth=1, edgecolor="w")

        for ax in g.axes.flat:
            # Make the grid horizontal instead of vertical
            ax.xaxis.grid(True, color='#e3e3e3')
            ax.yaxis.grid(True)

        g.axes.flat[0].set_xscale("log")
        g.axes.flat[0].set_xlim(3 * 1e2, 3 * 1e6)
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

        outfile = os.path.join(self.plot_path, 'sign2_node2vec_stats_CCs.png')
        plt.savefig(outfile, dpi=100)
        plt.close('all')
