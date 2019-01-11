"""Utility for plotting Chemical Checker data."""

import os
import json
import matplotlib
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

    def sign2_graph_stats(self):
        """Plot the stats for the neig1 deirived graph."""

        # plot selected stats
        stats = [
            "edges",
            "nodes",
            #"zeroNodes",
            "zeroInNodes",
            #"zeroOutNodes",
            #"nonZIODegNodes",
            "Connected Components",
            "Degree",
            "Weights"]
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
            stats_file = os.path.join(sign2.stats_path, "graph_stats.json")
            if not os.path.isfile(stats_file):
                self.__log.warn('Graph stats %s not found', stats_file)
                continue
            tmp = json.load(open(stats_file, 'r'))
            # prepare row
            for deg in degrees:
                row = dict()
                row.update(tmp)
                row.update({"dataset": ds})
                row.update({"Degree": tmp[deg]})
                df.loc[len(df)] = pd.Series(row)
            for conn in conncompo:
                row = dict()
                row.update(tmp)
                row.update({"dataset": ds})
                row.update({"Connected Components": tmp[conn]})
                df.loc[len(df)] = pd.Series(row)
            for wei in weights:
                row = dict()
                row.update(tmp)
                row.update({"dataset": ds})
                row.update({"Weights": tmp[wei]})
                df.loc[len(df)] = pd.Series(row)

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
        g.axes.flat[0].set_xlim(1e4, 1e8)
        g.axes.flat[0].set_xlabel("Edges")
        g.axes.flat[1].set_xscale("log")
        g.axes.flat[1].set_xlim(1e2, 1e7)
        g.axes.flat[1].set_xlabel("Nodes")
        g.axes.flat[-2].set_xscale("log")
        g.axes.flat[-1].set_xscale("log")
        # g.axes.flat[-1].set_xlim(1e1,1e3)
        sns.despine(left=True, bottom=True)

        outfile = os.path.join(self.plot_path, 'graph_stats.png')
        plt.savefig(outfile, dpi=100)
        plt.close()
