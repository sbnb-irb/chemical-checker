import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import pickle
import h5py
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from utils import utils
import collections
from tqdm import tqdm
from scipy.spatial import distance
from chemicalchecker.util.plot.util import coord_color, set_style

import os
import pickle
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, auc, matthews_corrcoef, recall_score


class ReportCard:

    def __init__(self,
                 validation_root,
                 model_root = None,
                 prediction_root = None,
                 libraries=None,
                 pchembl=None,
                 inchikeys=None,
                 cutoff = False,
                 significance_only = False,
                 all_identifiers = False,
                 overwrite = False):

        self.overwrite = overwrite
        self.all_identifiers = all_identifiers
        self.validation_root = os.path.abspath(validation_root)
        if model_root is not None:
            self.model_root = os.path.abspath(model_root)
        else:
            self.model_root = None

        if prediction_root is not None:
            self.prediction_root = os.path.abspath(prediction_root)
        else:
            self.prediction_root = None

        self.cutoff = cutoff
        self.pchembl = pchembl
        self.libraries = libraries

        self.inchikeys = inchikeys

        self.palette = {'Active': '#5A72B5',
                        'Inactive': '#EA5A49',
                        'Efficiency': '#B16BA8',
                        'Heatmap': sns.color_palette("Spectral", desat=0.95)}

        self.significance_only = significance_only
    def _calculate_cutoff(self):

        path = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)), "validation.pkl")
        with open(path, "rb") as f:
            val = pickle.load(f)
        co_5 = []
        for ytr, ypr in zip(val['test']['y_true'], val['test']['y_pred']):
            try:
                co_5 += [np.percentile(ypr[(ytr == 1) & (ypr[:, 0] < ypr[:, 1]), 1], 5)]
            except:
                co_5 += [0]
            if len(co_5) == 3:
                self.cutoff = np.median(co_5)


    def _get_training_data(self, activity=False):
        path = os.path.join(self.model_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)), "trained_data.pkl")
        with open(path, "rb") as f:
            td_ = pickle.load(f)
        if not activity:
            td = td_.inchikey
        else:
            td_ = td_.activity
            td_ = td_[td_ == 1]
            td = len(td_)
        if not activity:
            self.training_data = td
        else:
            self.training_data_active = td

    def _get_true_actives(self):
        p = None
        path = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)),
                            "validation.pkl")
        with open(path, "rb") as f:
            val = pickle.load(f)
        val = val['test']
        for ytr, ypr in zip(val['y_true'], val['y_pred']):
            if p is None:
                p = ypr[ytr == 1]
            else:
                p = np.vstack([p, ypr[ytr == 1]])
        return p

    def _get_predictions(self, l, repeated=None):
        if not os.path.exists(os.path.join(self.prediction_root, l, self.target + ".pkl")): return None
        with open(os.path.join(self.prediction_root, l, self.target + ".pkl"), "rb") as f:
            p = pickle.load(f)
        p = p[p[:, -1] == 1]
        if repeated is not None:
            p = p[repeated]
        return p[:, :2]

    def get_metrics(self):

        metrics = []
        path = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)),
                            "validation.pkl")
        if not os.path.exists(path): return None
        with open(path, "rb") as f:
            val = pickle.load(f)
        for i, (a, ytr, ypr) in enumerate(
            zip(val['test']['perfs']['aupr'], val['test']['y_true'], val['test']['y_pred'])):
            pre = precision_score(ytr, ypr[:, 0] < ypr[:, 1])
            mcc = matthews_corrcoef(ytr, ypr[:, 0] < ypr[:, 1])
            re = recall_score(ytr, ypr[:, 0] < ypr[:, 1])
            if self.cutoff:
                # if self.target not in self.cutoff: continue
                pre_aupr = precision_score(ytr, (ypr[:, 0] < ypr[:, 1]) & (ypr[:, 1] > self.cutoff))
                re_aupr = recall_score(ytr, (ypr[:, 0] < ypr[:, 1]) & (ypr[:, 1] > self.cutoff))
                metrics.append([self.target, i, a[1], pre, re, pre_aupr, re_aupr, mcc])
            else:
                metrics.append([self.target, i, a[1], pre, re, mcc])
        if self.cutoff:
            columns = ['Target', 'Fold', 'AUPR', 'Precision', 'Recall', 'Precision (--)',
                       'Recall (--)', 'MCC']

        else:
            columns = ['Target', 'Fold', 'AUPR', 'Precision', 'Recall', 'MCC']
        metrics = pd.DataFrame(metrics, columns=columns)
        return metrics.groupby('Target').mean().reset_index().melt(id_vars=['Target'], value_vars=columns[2:],
                                                                   var_name='Metric')

    def validation_prediction_plot(self, boxplot_ax):
        val = self._get_true_actives()
        df = pd.DataFrame(val, columns=['Inactive', 'Active'])
        large = []

        df['Type'] = 'Validation'

        l_ = []
        if self.libraries is not None:
            td_ = [i.split("-")[0] for i in self.training_data]
            for l in self.libraries:
                if self.inchikeys is not None:
                    k = [i.split("-")[0] for i in self.inchikeys[l]]
                    pred = self._get_predictions(l, repeated=~np.isin(k, td_))
                else:
                    pred = self._get_predictions(l, repeated=None)
                if pred is None: continue

                pred = pred[pred[:, 0] < pred[:, 1]]
                pred = pd.DataFrame(pred, columns=['Inactive', 'Active'])
                if pred.empty:
                    pred = pred.append({'Inactive': np.nan, 'Active': np.nan}, ignore_index=True)
                if l.upper().startswith("CC"): l = 'CC'
                num_predicted = len(pred) if pred.isna().all().Active == False else 0
                l_.append('{:s}\n({:d})'.format(l, num_predicted))
                pred['Type'] = '{:s}\n({:d})'.format(l, num_predicted)
                df = pd.concat([ df, pred ])
                if np.sum(pred['Inactive'] < pred['Active']) > 500:
                    large += ['{:s}\n({:d})'.format(l, num_predicted)]

        df['Predicted Activity'] = df['Inactive'] < df['Active']
        df['Predicted Activity'] = df['Predicted Activity'].map({True: 'Active', False: 'Inactive'})
        df = df.reset_index(drop=True)
        if large == []:
            sns.swarmplot(ax=boxplot_ax, data=df, x='Active', y='Type', hue='Predicted Activity',
                          hue_order=['Active', 'Inactive'], s=3,
                          palette=self.palette)
        else:
            df_ = df.copy()
            df_.loc[df.index[df_.Type.isin(large)], 'Active'] = np.nan
            df_.loc[df.index[df_.Type.isin(large)], 'Inactive'] = np.nan
            sns.swarmplot(ax=boxplot_ax, data=df_, x='Active', y='Type', hue='Predicted Activity',
                          hue_order=['Active', 'Inactive'], s=3,
                          palette={'Active': coord_color("C"), 'Inactive': coord_color("A")})
        sns.boxplot(ax=boxplot_ax, data=df, x='Active', y='Type', boxprops={'facecolor': 'none'}, fliersize=3,
                    width=.55, zorder=1)
        boxplot_ax.set_yticklabels(boxplot_ax.get_yticklabels(), rotation=90, ha='center', va='center')
        boxplot_ax.set_ylabel('')
        boxplot_ax.set_xlabel('Active Probability')
        if self.cutoff:
            boxplot_ax.axvline(self.cutoff, color='dimgrey', linestyle=':')
        boxplot_ax.tick_params(axis='y', which='major', pad=15)
        boxplot_ax.set_xlim(-0.015, 1)

    def metrics_plot(self, metrics_ax):
        metrics = self.get_metrics()
        norm = plt.Normalize(0, 1)
        sns.heatmap(ax=metrics_ax, data=metrics.iloc[:, 1:].set_index('Metric'), cmap=self.palette['Heatmap'],
                    norm=norm, linewidths=.5,
                    annot=True, cbar=False, fmt='.2f')
        metrics_ax.set_yticklabels(metrics_ax.get_yticklabels(), rotation=0, ha='right', va='center', weight='semibold')
        metrics_ax.set_ylabel('')
        metrics_ax.set_xlabel('')
        metrics_ax.set_xticklabels([], )
        metrics_ax.tick_params(axis=u'both', which=u'both', length=0)

    def validity(self, ax):
        path = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)))
        valid_path = os.path.join(path, "validation.pkl")
        if not os.path.exists(valid_path):
            raise Exception("Validation path %s does not exist" % valid_path)

        with open(valid_path, "rb") as f:
            valid = pickle.load(f)
        ax.plot([0, 1], [0, 1], color='lightgrey')
        labelA = "Active"
        labelI = "Inactive"
        ina_all = []
        act_all = []
        glb_all = []
        for y_pred, y_true in zip(valid["test"]["y_pred"], valid["test"]["y_true"]):
            A_true = set(np.where(y_true == 1)[0])
            I_true = set(np.where(y_true == 0)[0])
            cls = np.arange(0, 1, 0.01)
            glb = []
            act = []
            ina = []
            cls_glb = []
            cls_a = []
            cls_i = []
            for cl in cls:
                # epsilon = 1 - cl
                # A = set(np.where(y_pred[:, 1] > epsilon)[0])
                # act += [len(A.intersection(A_true)) / len(A_true)]
                # cls_a += [cl]
                # I = set(np.where(y_pred[:, 0] > epsilon)[0])
                # ina += [len(I.intersection(I_true)) / len(I_true)]
                # cls_i += [cl]

                epsilon = 1 - cl
                A = set(np.where(y_pred[:, 1] > epsilon)[0])
                act += [(len(A_true) - len(A.intersection(A_true))) / len(A_true)]
                cls_a += [epsilon]
                I = set(np.where(y_pred[:, 0] > epsilon)[0])
                ina += [(len(I_true) - len(I.intersection(I_true))) / len(I_true)]
                cls_i += [epsilon]

                # glb += [((len(A_true) - len(A.intersection(A_true))) + (len(I_true) - len(I.intersection(I_true)))) / (len(A_true)+len(I_true))]
                # cls_glb += [epsilon]


            ax.plot(cls_i, ina, label='_nolegend_', color=coord_color("A"), alpha = 0.4, linestyle = '--')
            ax.plot(cls_a, act, label='_nolegend_', color=coord_color("C"), alpha = 0.4, linestyle = '--')
            ina_all.append(ina)
            act_all.append(act)
            # glb_all.append(glb)
            labelI = '_nolegend_'
            labelA = '_nolegend_'

        ina_all = np.mean(np.asarray(ina_all), axis=0)
        act_all = np.mean(np.asarray(act_all), axis=0)
        # glb_all = np.mean(np.asarray(glb_all), axis=0)

        ax.plot(cls_i, ina_all, label="Inactive", color=coord_color("A"))
        ax.plot(cls_a, act_all, label="Active", color=coord_color("C"))
        # ax.plot(cls_glb, glb_all, label="Overall", color=coord_color("B"))

        ax.legend()
        ax.set_xlabel("Significance")
        ax.set_ylabel("Error rate")
        ax.set_title("Validity")
        ax.set_ylim(-0.015, 1)
        ax.set_xlim(-0.015, 1)

    def efficiency(self, ax):
        """Fraction of single-class predictions that are correct."""

        path = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)))
        valid_path = os.path.join(path, "validation.pkl")
        if not os.path.exists(valid_path):
            raise Exception("Validation path %s does not exist" % valid_path)

        with open(valid_path, "rb") as f:
            valid = pickle.load(f)
        for y_pred, y_true in zip(valid["test"]["y_pred"], valid["test"]["y_true"]):
            A_true = set(np.where(y_true == 1)[0])
            I_true = set(np.where(y_true == 0)[0])
            cls_ = np.arange(0, 1, 0.01)
            eff = []
            cls = []
            for cl in cls_:
                epsilon = 1 - cl
                A = set(np.where(y_pred[:, 1] > epsilon)[0])
                I = set(np.where(y_pred[:, 0] > epsilon)[0])
                if (len(A) + len(I)) == 0: continue
                corr = len(A.intersection(A_true)) + len(I.intersection(I_true))
                eff += [corr / (len(A) + len(I))]
                cls += [epsilon]
            ax.plot(cls, eff, color=coord_color("B"))

        ax.set_xlabel("Significance")
        ax.set_ylabel("Correct predictions")
        ax.set_title("Efficiency")
        ax.set_ylim(-0.015, 1)
        ax.set_xlim(-0.015, 1)

    def _canvas(self, opath=None, genename=None, show_known_actives=False):
        if opath is not None:
            if opath == 'validation':
                opath = os.path.join(self.validation_root, self.target, "pchembl_{:d}".format(int(self.pchembl * 100)), "targetcard")
            else:
                opath = os.path.join(opath, "{:s}".format("_".join(genename[self.target].split("/") if genename is not None else self.target)))
            if not self.overwrite:
                if os.path.exists(opath):
                    return None
        if self.cutoff:
            self._calculate_cutoff()
        if self.model_root:
            self._get_training_data()
            self._get_training_data(activity=True)
        fig = plt.figure(constrained_layout=True, dpi=200, figsize=(9, 5.2))

        gs = GridSpec(4, 8, figure=fig)
        metrics_ax = fig.add_subplot(gs[:, :1])
        boxplot_ax = fig.add_subplot(gs[:, 3:])
        validity_ax = fig.add_subplot(gs[:2, 1:3])
        efficiency_ax = fig.add_subplot(gs[2:, 1:3])

        self.metrics_plot(metrics_ax)
        self.validation_prediction_plot(boxplot_ax)
        self.validity(validity_ax)
        self.efficiency(efficiency_ax)

        if self.significance_only:
            validity_ax.set_xlim(0.7,1)
            validity_ax.set_ylim(0.6, 1)
            efficiency_ax.set_xlim(0.7,1)

        if self.all_identifiers:
            title = " / ".join([v[self.target] for v in self.all_identifiers.values() if self.target in v])
            if title != '':
                title = "{:s} / {:s}".format(self.target, title)
            else:
                title = "{:s}".format(self.target)
        else:
            title = "{:s}".format(genename[self.target] if genename is not None else self.target)
        if show_known_actives:
            title = title + "\n{:d} Active compounds".format(self.training_data_active)

        plt.suptitle(title, va='center', ha='center')

        if opath is not None:
            plt.savefig(opath, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            return fig


class TargetReportCard(ReportCard):
    def __init__(self,
                 opath=None,
                 genename=None,
                 show_known_actives=False,
                 **kwargs):
        ReportCard.__init__(self, **kwargs)
        self.opath = opath
        self.genename = genename
        self.show_known_actives = show_known_actives
    def run(self, targets=None, pchembl = None):
        # TODO: Add overwrite
        if targets is not None: # TODO: Fix this for new process
            if type(targets) == str:
                targets = [targets]
            elif type(targets) == list:
                pass
        else:
            for target in os.listdir(self.validation_root):
                for pc in os.listdir(os.path.join(self.validation_root, target)):
                    if pchembl is not None:
                        if "pchembl_{:d}".format(int(pchembl * 100)) != pc: continue

                    path = os.path.join(self.validation_root, target, pc,"validation.pkl")
                    if not os.path.exists(path): continue

                    self.target = target
                    self.pchembl = int(pc.split("_")[-1])/100
                    self._canvas(opath=self.opath, genename=self.genename, show_known_actives=self.show_known_actives)

        del self.target
