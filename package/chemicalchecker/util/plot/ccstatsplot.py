"""CC statistics plots for CC web page."""
import os
import copy
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import patches
from .util import canvas, cc_grid, cc_colors, homogenous_ticks, set_style, cc_coords_name

from chemicalchecker.util import logged

set_style()


@logged
class CCStatsPlot(object):
    """CCStatsPlot class."""

    def __init__(self, cc, width=30, height=30, dpi=70, transparent=True,
                 save=True, save_format='png', save_dir='./'):
        """Initialize a CCStatsPlot instance.

        The plotter works on data precomputed mainly using
        :mod:`~chemicalchecker.core.diagnostics`.

            Args:
                cc (ChemicalChecker): A ChemicalChecker object.
        """
        self.cc = cc
        self.width = width
        self.height = height
        self.dpi = dpi
        self.transparent = transparent
        self.save_format = save_format
        self.save_dir = save_dir

    def plot_all(self):
        """Run all plots reported in the 'available' table."""
        # TODO: take kwargs dict for each plot
        for method, _ in self.available().values:
            self.__log.info('Plotting: %s' % method)
            eval('self.%s()' % method)

    def available(self):
        """Resume of possible plots."""
        d = {
            "matrices": "Number of molecules and signature lengths in each of the 25 Chemical Checker datasets. Signature lengths can be read as a measure of complexity or sparsity of the data.",
            "moa_validations": "Chemical Checker datasets correlates with mechanisms of action (MoA). The receiver-operating characteristic (ROC) curves measure how similar molecules tend to share MoA. Note that the almost-perfect performance in the 'Mechanism of action' dataset is trivial.",
            "correlations": "Degree of correlation between data types measured as the ability to use one signature (x-axis) to recover neighbors defined with another signature (y-axis). Red denotes high correlation and blue denotes low correlation.",
        }
        R = []
        for k in sorted(d.keys()):
            R += [(k, d[k])]
        df = pd.DataFrame(R, columns=["method", "description"])
        return df

    def save(self, fig):
        filename = '%s.%s' % (inspect.stack()[1][3], self.save_format)
        fig.savefig(os.path.join(self.save_dir, filename),
                    bbox_inches='tight', transparent=self.transparent)

    def matrices(self, cctype='sign0', molset='full', max_mols=6.5,
                  max_feat=4.5):
        # create the grid
        fig, grid = canvas(width=self.width, height=self.height, dpi=self.dpi)
        axes = cc_grid(fig, grid, legend_out=False, cc_space_names=True,
                       hspace=0.2, wspace=0.2,
                       shared_ylabel='Molecules (log10)',
                       shared_xlabel='Variables (log10)')
        # check maximum sizes
        dims = dict()
        for ds in self.cc.datasets_exemplary():
            try:
                nr_mol, nr_feat = self.cc.metadata[
                    'dimensions'][molset][ds][cctype]
            except Exception as ex:
                self.__log.error('Cannot fetch cc.metadata: %s' % str(ex))
                continue
            max_mols = np.max([np.log10(nr_mol), max_mols])
            max_feat = np.max([np.log10(nr_feat), max_feat])
            dims[ds] = (nr_mol, nr_feat)
        # plot a rectangle for each space
        for ax, ds in zip(axes, self.cc.datasets_exemplary()):
            color = cc_colors(ds, lighness=.3, alternate=True, dark_first=True)
            if ds not in dims:
                continue
            nr_mol, nr_feat = dims[ds]
            rect = patches.Rectangle(
                (0, 0), np.log10(nr_feat), np.log10(nr_mol),
                facecolor=color, edgecolor='k', linewidth=1, alpha=0.9)
            ax.add_patch(rect)
            ax.set_ylim(0, max_mols)
            ax.set_xlim(0, max_feat)
            ax.set_aspect('auto')
            ax.set_yticks(range(int(np.ceil(max_mols))))
            ax.set_xticks(range(int(np.ceil(max_feat))))
            if 'E' in ds[:2]:
                ax.set_xticklabels(range(int(np.ceil(max_feat))))
            if '1' in ds[:2]:
                ax.set_yticklabels(range(int(np.ceil(max_mols))))
        # save or return
        if self.save:
            self.save(fig)
        else:
            return fig

    def moa_validations(self, cctype='sign1'):
        fig, grid = canvas(width=self.width, height=self.height, dpi=self.dpi)
        axes = cc_grid(fig, grid, legend_out=False, cc_space_names=True,
                       hspace=0.2, wspace=0.2,
                       shared_ylabel='True positive rate',
                       shared_xlabel='False positive rate')
        for ax, ds in zip(axes, self.cc.datasets_exemplary()):
            color = cc_colors(ds, lighness=.3, alternate=True, dark_first=True)
            sign = self.cc.signature(ds, cctype)
            diag = sign.diagnosis(ref_cc=self.cc)
            try:
                diag.plotter.moa_roc(title=False, color=color,
                                     xylabels=False, ax=ax)
            except Exception as ex:
                self.__log.error('Cannot fetch moa_roc: %s' % str(ex))
                continue
            ax.set_aspect('auto')
            ax.set_xticks([0,0.5,1])
            ax.set_xticklabels(['0','.5','1'])
            ax.set_yticks([0,0.5,1])
            ax.set_yticklabels(['0','.5','1'])
            if 'E' not in ds[:2]:
                ax.set_xticklabels('')
            if '1' not in ds[:2]:
                ax.set_yticklabels('')
        # save or return
        if self.save:
            self.save(fig)
        else:
            return fig

    def correlations(self, cctype='sign1'):
        rocauc_matrix = np.full((25, 25), np.nan)
        for row, ds1 in enumerate(self.cc.datasets_exemplary()):
            sign = self.cc.signature(ds1, cctype)
            diag = sign.diagnosis(ref_cc=self.cc)
            try:
                res = diag._load_diagnosis_pickle('across_roc.pkl')
            except Exception as ex:
                self.__log.error('Cannot fetch across_roc: %s' % str(ex))
                continue
            for col, ds2 in enumerate(self.cc.datasets_exemplary()):
                if ds2 not in res or res[ds2] is None:
                    continue
                rocauc_matrix[row, col] = res[ds2]['auc']
        fig, grid = canvas(width=self.width, height=self.height, dpi=self.dpi)
        ax = fig.add_subplot(grid[:])
        cmap = copy.copy(cm.get_cmap("RdYlBu_r"))
        cmap.set_bad(".5")
        names = [cc_coords_name(ds[:2]) for ds in self.cc.datasets_exemplary()]
        sns.heatmap(
            rocauc_matrix, vmin=0.5, vmax=1, square=True,
            mask=np.isnan(rocauc_matrix), xticklabels=names, yticklabels=names,
            linewidth=1, linecolor='.5', cbar=False, annot=False, cmap=cmap,
            ax=ax)
        for i in range(0, 30, 5):
            ax.hlines(i, 0, 25, color='k', lw=2,
                      capstyle='round').set_clip_on(False)
            ax.vlines(i, 0, 25, color='k', lw=2,
                      capstyle='round').set_clip_on(False)
        # save or return
        if self.save:
            self.save(fig)
        else:
            return fig
