import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Rectangle
from scipy import stats
import numpy as np
from chemicalchecker.util.plot.style.util import coord_color, set_style
from chemicalchecker.util import logged

set_style()

@logged
class OneConformalClassifierPlot(object):
    
    def __init__(self, path, significance = 0.2):
        """Initialize class
        
        Args:
            path(str): Path to the folder where results are stored.
            significance(float): Significance threshold, according to the conformal classification scheme (default=0.2).
        """
        
        self.path = os.path.abspath(path)
        self.valid_path = os.path.join(self.path, "validation.pkl")
        if not os.path.exists(self.valid_path):
            raise Exception("Validation path %s does not exist" % self.valid_path)
        self.outpath = os.path.join(self.path, "plots")
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        with open(self.valid_path, "rb") as f: 
            valid = pickle.load(f)
        if type(valid) is not dict:
            self.valid = valid.as_dict()
        else:
            self.valid = valid
        self.significance = significance
        self.__log.info("Conformal classifier plots. Results will be stored in %s" % self.outpath)
    
    @staticmethod
    def _get_j(label):
        assert label in set(["Active", "Inactive"]), "Only Active and Inactive are valid labels"
        if label == "Active":
            j = 1
        else:
            j = 0
        return j
    
    def find_significance_threshold(self, label):
        j = self._get_j(label)
        vals = []
        for spl in range(0, self.valid["n_splits"]):
            y_true = self.valid["test"]["y_true"][spl]
            y_pred = self.valid["test"]["y_pred"][spl][:,j]
            fpr, tpr, thr = roc_curve(y_true, y_pred)
            bacc = 0.5*((1 - fpr) + tpr)
            idx = np.argmax(bacc)
            vals += [thr[idx]]
        return np.mean(vals)
    
    def ranking_metrics(self, ax, label):
        self.__log.debug("Ranking metrics plot")
        j = self._get_j(label)
        mets = ["auroc", "aupr", "bedroc"]
        if self.valid["is_ensemble"]:
            v = []
            x = None
            for spl in range(0, self.valid["n_splits"]):
                datasets = self.valid["datasets"]
                colors = [coord_color(ds) for ds in datasets]
                perfs = self.valid["ens_test"]["perfs"]
                for i, k in enumerate(mets):
                    v += [perfs[k][spl][j,:]]
                    if x is None:
                        x = np.array([i for _ in range(0, len(v[-1]))])
            v = np.array(v)
            v = np.mean(v, axis = 0)
            noise = np.random.normal(0, 0.05, x.shape)
            x = x + noise
            ax.scatter(x, v, color = colors)
        perfs = self.valid["test"]["perfs"]
        y = []
        for spl in range(0, self.valid["n_splits"]):
            x = []
            y_ = []
            for i, k in enumerate(mets):
                y_ += [perfs[k][spl][j]]
                x += [i]
            y += [y]
        y = np.array(y)
        y = np.mean(y, axis = 0)
        ax.scatter(x, y, color = "grey")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in mets])
        ax.set_ylabel("Performance")
        ax.set_title("Ranking metrics")
        ax.grid()
            
    def classification_metrics(self, ax, label, significance=None):
        self.__log.debug("Classification metrics plot")    
        mets = [
            ("Kappa", metrics.cohen_kappa_score),
            ("MCC"  , metrics.matthews_corrcoef),
            ("Prec.", metrics.precision_score),
            ("Rec." , metrics.recall_score)
        ]
        j = self._get_j(label)
        if not significance:
            significance = self.find_significance_threshold(label)
        if self.valid["is_ensemble"]:
            datasets = self.valid["datasets"]
            colors = [coord_color(ds) for ds in datasets]
            y_true   = self.valid["ens_test"]["y_true"]
            y_pred   = self.valid["ens_test"]["y_pred"]
            for i, k in enumerate(mets):
                m = k[1]
                Y = y_pred[:,j,:]
                v = np.zeros(Y.shape[1])
                for l in range(0, Y.shape[1]):
                    y = np.zeros(Y.shape[0])
                    y[Y[:,l] > significance] = 1
                    v[l] = m(y_true, y)
                x = np.array([i for _ in range(0, len(v))])
                noise = np.random.normal(0, 0.05, x.shape)
                x = x + noise
                ax.scatter(x, v, color = colors)
        y_pred = self.valid["test"]["y_pred"][:,j]
        x = []
        v = []
        y = np.zeros(y_pred.shape)
        y[y_pred > significance] = 1
        for i, k in enumerate(mets):
            m = k[1]
            v += [m(y_true, y)]
            x += [i]
        ax.scatter(x, v, color = "grey")
        ax.set_xticks(x)
        ax.set_xticklabels([k[0] for k in mets])
        ax.set_ylabel("Performance")
        ax.set_title("PS %.2f" % significance)
        ax.grid()
        
    def pvalue_distributions(self, ax, label):
        self.__log.debug("P-values plot")
        j = self._get_j(label)
        if label == "Active":
            lab  = "Active"
            lab_ = "Inactive"
            col  = coord_color("C")
            col_ = coord_color("A")
        else:
            lab  = "Inactive"
            lab_ = "Active"
            col  = coord_color("A")
            col_ = coord_color("C")
        y_pred = self.valid["test"]["y_pred"][:,j][self.valid["test"]["y_true"]  == j]
        y_pred_ = self.valid["test"]["y_pred"][:,j][self.valid["test"]["y_true"] != j]
        x = np.arange(0, 1, 0.01)
        density = stats.kde.gaussian_kde(y_pred)
        density_ = stats.kde.gaussian_kde(y_pred_)
        if label == "Active":
            ax.plot(x, density(x), label = lab, color = col)
            ax.plot(x, density_(x), label = lab_, color = col_)
        else:
            ax.plot(x, density_(x), label = lab_, color = col_)
            ax.plot(x, density(x), label = lab, color = col)
        ax.set_xlabel("PS")
        ax.set_ylabel("Density")
        ax.set_title("%s set" % label)
        ax.legend()
        ax.grid()
    
    def roc_curve(self, ax, label):
        self.__log.debug("ROC curve")
        j = self._get_j(label)
        y_true_tr = np.abs(self.valid["train"]["y_true"] - (1-j))
        y_pred_tr = self.valid["train"]["y_pred"][:,j]
        fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
        auc_tr = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = "Train", color = coord_color("C"))
        y_true_tr = np.abs(self.valid["test"]["y_true"] - (1-j))
        y_pred_tr = self.valid["test"]["y_pred"][:,j]
        fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
        auc_ts = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = "Test", color = coord_color("A"))
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC %.2f / %.2f" % (auc_tr, auc_ts))
        ax.legend()
        ax.grid()

    def pr_curve(self, ax, label):
        self.__log.debug("PR curve")
        j = self._get_j(label)
        y_true_tr = np.abs(self.valid["train"]["y_true"] - (1-j))
        y_pred_tr = self.valid["train"]["y_pred"][:,j]
        pre, rec, _ = precision_recall_curve(y_true_tr, y_pred_tr)
        auc_tr = auc(rec, pre)
        ax.plot(rec, pre, label = "Train", color = coord_color("C"))
        y_true_ts = np.abs(self.valid["test"]["y_true"] - (1-j))
        y_pred_ts = self.valid["test"]["y_pred"][:,j]
        pre, rec, _ = precision_recall_curve(y_true_ts, y_pred_ts)
        auc_ts = auc(rec, pre)
        ax.plot(rec, pre, label = "Test", color = coord_color("A"))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR %.2f / %.2f" % (auc_tr, auc_ts))
        ax.legend()
        ax.grid()
        
    def confidence_level(self, ax):
        self.__log.debug("Confidence level plot")
        y_pred = self.valid["test"]["y_pred"]
        y_true = self.valid["test"]["y_true"]
        As = []
        Is = []
        Bs = []
        Ns = []
        cls = np.arange(0,1,0.01)
        for cl in cls:
            epsilon = 1 - cl
            A = set(np.where(y_pred[:,1] > epsilon)[0])
            I = set(np.where(y_pred[:,0] > epsilon)[0])
            B = A.intersection(I)
            A = A.difference(B)
            I = I.difference(B)
            N = set([i for i in range(0, len(y_true))]).difference(A.union(I.union(B)))
            As += [len(A)]
            Is += [len(I)]
            Bs += [len(B)]
            Ns += [len(N)]
        ax.plot(cls, As, label = "Active", color = coord_color("C"))
        ax.plot(cls, Is, label = "Inactive", color = coord_color("A"))
        ax.plot(cls, Bs, label = "Both", color = coord_color("B"))
        ax.plot(cls, Ns, label = "Null", color = "grey")
        ax.legend()
        ax.set_ylabel("Compounds")
        ax.set_xlabel("Confidence level")
        ax.set_title("Decisions")
        ax.grid()
        
    def validity(self, ax):
        self.__log.debug("Validity plot")
        y_pred = self.valid["test"]["y_pred"]
        y_true = self.valid["test"]["y_true"]
        A_true = set(np.where(y_true == 1)[0])
        I_true = set(np.where(y_true == 0)[0])
        cls = np.arange(0,1,0.01)
        glb = []
        act = []
        ina = []
        cls_a = []
        cls_i = []
        for cl in cls:
            epsilon = 1 - cl
            A = set(np.where(y_pred[:,1] > epsilon)[0])
            act += [len(A.intersection(A_true))/len(A_true)]
            cls_a += [cl]
            I = set(np.where(y_pred[:,0] > epsilon)[0])
            ina += [len(I.intersection(I_true))/len(I_true)]
            cls_i += [cl]
        ax.plot(cls_i, ina, label = "Inactive", color = coord_color("A"))
        ax.plot(cls_a, act, label = "Active", color = coord_color("C"))
        ax.legend()
        ax.set_xlabel("Confidence level")
        ax.set_ylabel("Validity")
        ax.set_title("Validity")
        ax.grid()
        
    def efficiency(self, ax):
        self.__log.debug("Efficiency plot")
        """Fraction of single-class predictions that are correct."""
        y_pred = self.valid["test"]["y_pred"]
        y_true = self.valid["test"]["y_true"]
        A_true = set(np.where(y_true == 1)[0])
        I_true = set(np.where(y_true == 0)[0])
        cls_ = np.arange(0,1,0.01)
        eff = []
        cls = []
        for cl in cls_:
            epsilon = 1 - cl
            A = set(np.where(y_pred[:,1] > epsilon)[0])
            I = set(np.where(y_pred[:,0] > epsilon)[0])
            if (len(A) + len(I)) == 0: continue
            corr = len(A.intersection(A_true)) + len(I.intersection(I_true))
            eff += [corr / (len(A) + len(I))]
            cls += [cl]
        ax.plot(cls, eff, color = coord_color("B"))
        ax.set_xlabel("Confidence level")
        ax.set_ylabel("Efficiency")
        ax.set_title("Efficiency")
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.grid()
               
    def both_pvalues(self, ax):
        self.__log.debug("Both p-values plot")
        y_pred = self.valid["test"]["y_pred"]
        x = y_pred[:,0]
        y = y_pred[:,1]
        xy = np.vstack([x,y])
        z = stats.kde.gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]    
        ax.scatter(x, y, c=z, edgecolor = "", cmap = "Spectral")
        ax.set_xlabel("PS Inactive")
        ax.set_ylabel("PS Active")
        ax.set_title("Class. PS")
        ax.grid()
        
    def calibration(self, ax, label):
        self.__log.debug("Calibration plot")
        if label == "Active":
            color = coord_color("C")
        else:
            color = coord_color("A")
        j = self._get_j(label)
        y_pred = self.valid["test"]["y_pred"][:,j][self.valid["test"]["y_true"] == j]
        ax.plot([(i+1) for i in range(0, len(y_pred))], sorted(y_pred), color = color)
        ax.set_title("Calibr. %s" % label)
        ax.set_xlabel("Test set compounds")
        ax.set_ylabel("PS")
        ax.set_ylim(0,1)
        ax.grid()
        
    def heatmap(self, ax, label, max_samples = 1000, cmap = "YlGnBu", cluster = False):
        self.__log.debug("Heatmap")
        # Remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)        
        # Fetch data        
        j = self._get_j(label)
        y_pred = self.valid["ens_test"]["y_pred"][:,j,:]
        mask = self.valid["ens_test"]["y_true"] == j
        y_pred = y_pred[mask]
        datasets = self.valid["datasets"]
        # Main matrix
        M = np.zeros((len(datasets), y_pred.shape[0]))
        for i in range(0, M.shape[0]):
            for j in range(0, M.shape[1]):
                M[i,j] = y_pred[j,i]
        if cluster:
            idxs = dendrogram(linkage(M.T), no_plot=True)["leaves"]
        else:
            idxs = np.argsort(self.valid["test"]["y_pred"][mask,self._get_j(label)])
        cmap = plt.cm.get_cmap(cmap)
        ax.imshow(M[:,idxs], aspect = "auto", cmap = cmap, vmin=0, vmax=1)
        ax.grid(False)
        r = Rectangle((-0.5, -0.5), y_pred.shape[0], len(datasets), fill = False, edgecolor = "black")
        ax.add_patch(r)        
        # Get dimensions
        abs_w  = y_pred.shape[0]
        abs_h  = len(datasets)
        xlim_r = (-0.5, abs_w - 0.5)
        ylim_r = (abs_h - 0.5, -0.5)
        w, h = ax.get_figure().get_size_inches()
        bars_width_r = 0.05
        pads_width_r = bars_width_r*0.8
        # Colorbar
        cbar_thick = abs_h*bars_width_r
        cpad_thick = abs_h*pads_width_r
        if h > w:
            cbar_thick *= (w/h)
            cpad_thick *= (w/h)
        gran = 20
        w_prop = 0.5
        interv = y_pred.shape[0]/gran*w_prop
        colors = cmap(np.arange(0,1,1/gran))
        anchors = np.arange((1 - w_prop)*(xlim_r[1] - xlim_r[0]), xlim_r[1], interv)
        for i, anch in enumerate(anchors):
            c = colors[i]
            r = Rectangle((anch, ylim_r[1] - (cpad_thick + cbar_thick)), interv, cbar_thick, color = c)
            ax.add_patch(r)
        r = Rectangle(((1 - w_prop)*(xlim_r[1] - xlim_r[0]), ylim_r[1] - (cpad_thick + cbar_thick)), y_pred.shape[0]*w_prop, cbar_thick, fill = False, edgecolor = "black")
        ax.add_patch(r)
        ylim = (ylim_r[0]*1.01, (ylim_r[1] - (cpad_thick + cbar_thick))*1.01)
        # Datasets colors
        dbar_thick = abs_w*bars_width_r*2
        dpad_thick = abs_w*pads_width_r
        if w > h:
            dbar_thick *= (h/w)
            dpad_thick *= (h/w)
        for i, ds in enumerate(datasets):
            r = Rectangle((xlim_r[0] - (dbar_thick + dpad_thick), i - 0.5), dbar_thick, 1, color = coord_color(ds))
            ax.add_patch(r)
        r = Rectangle((xlim_r[0] - (dbar_thick + dpad_thick), -0.5), dbar_thick, len(datasets), fill = False, edgecolor = "black")
        ax.add_patch(r)
        xlim = ((xlim_r[0] - (dbar_thick + dpad_thick))*1.01, xlim_r[1]*1.01)
        ax.text(anchors[0] - dpad_thick, ylim_r[1] - (cpad_thick + cbar_thick/2),
                "PS", va = "center", ha = "right")
        # Frames etc.
        ax.set_yticks([], [])
        xticks = [x for x in ax.get_xticks() if x >= 0]
        ax.set_xticks(xticks)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_ylabel("Individual models")
        ax.set_xlabel("Compounds")
        
    def sorted_by_precision(self, ax, label, significance=None):
        self.__log.debug("Sorted-by-precision plot")
        j = self._get_j(label)
        y_true = valid["ens_test"]["y_true"]
        y_pred = valid["ens_test"]["y_pred"][:,j,:]
        y = np.zeros(y_pred.shape)
        y[y_pred > significance] = 1
        prcs = []
        for l in range(0, y.shape[1]):
            prcs += [metrics.precision_score(y_true, y[:,l])]
        idxs = np.argsort(-np.array(prcs))
        datasets = np.array(self.valid["datasets"])[idxs]
        y = np.zeros(y_true.shape)
        prcs = []
        recs = []
        for idx in idxs:
            y_ = y_pred[:, idx]
            y[y_ > significance] = 1
            prcs += [metrics.precision_score(y_true, y)]
            recs += [metrics.recall_score(y_true, y)]
        prcs = np.array(prcs)
        recs = np.array(recs)
        ax.plot([i+1 for i in range(len(prcs))], prcs, label = "Precision", color = coord_color("C1", False))
        ax.plot([i+1 for i in range(len(recs))], recs, label = "Recall", color = coord_color("A1", False))
        for i, ds in enumerate(datasets):
            i += 1
            r = Rectangle((i-0.5, -0.1), 1, 0.1, color = coord_color(ds, False))
            ax.add_patch(r)
        ax.legend()
        ax.set_ylim(-0.1, 1)
        ax.set_xlim(0.5, len(datasets)+0.5)
        ax.set_title("PS %.2f" % significance)
        ax.set_ylabel("Performance")
        ax.set_xlabel("Datasets (by prec.)")
        ax.grid()
        
    def _canvas(self):
        self.__log.debug("Canvas")
        fig, axs = plt.subplots(4,4, figsize = (10,10))
        #self.classification_metrics(axs[0,0], "Active")
        self.ranking_metrics(axs[0,1], "Active")
        self.roc_curve(axs[0,2], "Active")
        self.pr_curve(axs[0,3], "Active")
        self.pvalue_distributions(axs[1,0], "Active")
        self.pvalue_distributions(axs[1,1], "Inactive")
        # Calibration
        self.calibration(axs[1,2], "Active")
        self.calibration(axs[1,3], "Inactive")
        # Others
        self.confidence_level(axs[2,0])
        self.both_pvalues(axs[2,1])
        self.validity(axs[2,2])
        self.efficiency(axs[2,3])
        self.heatmap(axs[3,0], "Active")
        self.sorted_by_precision(axs[3,1], label = "Active", significance = 0.9)
        self.sorted_by_precision(axs[3,2], label = "Active", significance = 0.75)
        self.sorted_by_precision(axs[3,3], label = "Active", significance = 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outpath, "validation.png"), dpi=300)

    def canvas(self):
        try:
            self._canvas()
        except:
            self.__log.warning("Plots were not successful")


class OneClassifierPlot:
    """Do plots for one single TargetMate model"""

    def __init__(self, valid, **kwargs):
        """
        Args:
            valid(Validation): A validation instance.
        """
        if type(valid) is not dict:
            self.valid = valid.as_dict()
        
    def scores_distribution(self, ax):
        """Distribution of scores"""
        y_pred_tr = self.valid["train"]["y_pred"]
        y_pred_ts = self.valid["test" ]["y_pred"]
        density = stats.kde.gaussian_kde(y_pred_tr)
        x = np.arange(0, 1, 0.01)
        ax.plot(x, density(x), color = "black")
        density = stats.kde.gaussian_kde(y_pred_ts)
        x = np.arange(0, 1, 0.01)
        ax.plot(x, density(x), color = "red")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        pos = np.sum(p["perf_train"]["y_true"]) + np.sum(p["perf_test"]["y_true"])
        neg = len(p["perf_train"]["y_true"]) + len(p["perf_test"]["y_true"]) - pos 
        ax.set_title("Pos. %d / Neg. %d" % (pos, neg))

    def individual_performances(ax, perfs, metric):
        keys = sorted([k for k in perfs.keys() if k != "MetaPred"]) + ["MetaPred"]
        tr = []
        ts = []
        for k in keys:
            p = perfs[k]
            tr += [p["perf_train"][metric][0]]
            ts += [p["perf_test"][metric][0]]
        ax.scatter([i for i in range(0, len(keys))], tr, color = "white", edgecolor = ["black"]*(len(tr) - 1) + ["red"])
        ax.scatter([i for i in range(0, len(keys))], ts, color = ["black"]*(len(tr) - 1) + ["red"])
        ax.set_ylabel(metric.upper())
        ax.set_title("%.2f / Avg. %.2f / Best %.2f" % (ts[-1], np.mean(ts[:-1]), np.max(ts[:-1])))
        
    def rocs(ax, perfs):
        p = perfs["MetaPred"]
        y_true_tr = p["perf_train"]["y_true"]
        y_true_ts = p["perf_test"]["y_true"]
        y_pred_tr = p["perf_train"]["y_pred"]
        y_pred_ts = p["perf_test"]["y_pred"]
        fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
        ax.plot([0] + list(fpr) + [1], [0] + list(tpr) + [1], color = "black")
        fpr, tpr, _ = roc_curve(y_true_ts, y_pred_ts)
        ax.plot([0] + list(fpr) + [1], [0] + list(tpr) + [1], color = "red")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("Train %.2f / Test %.2f" % (p["perf_train"]["auroc"][0], p["perf_test"]["auroc"][0]))
            
    def precrecs(ax, perfs):
        p = perfs["MetaPred"]
        y_true_tr = p["perf_train"]["y_true"]
        y_true_ts = p["perf_test"]["y_true"]
        y_pred_tr = p["perf_train"]["y_pred"]
        y_pred_ts = p["perf_test"]["y_pred"]
        pre, rec, _ = precision_recall_curve(y_true_tr, y_pred_tr)
        ax.plot(list(rec), list(pre), color = "black")
        pre, rec, _ = precision_recall_curve(y_true_ts, y_pred_ts)
        ax.plot(list(rec), list(pre), color = "red")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Train %.2f / Test %.2f" % (p["perf_train"]["aupr"][0], p["perf_test"]["aupr"][0]))
        
    def applicability(ax, perfs, ad_data, col):
        if col == 0:
            name = "Standard deviation"
        if col == 1:
            name = "Precision"
        if col == 2:
            name = "Weight"
        x = np.array(ad_data[:, col])
        y = np.array(perfs["MetaPred"]["perf_test"]["y_pred"])
        xy = np.vstack([x,y])
        z = stats.kde.gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]    
        ax.scatter(x, y, c=z, edgecolor = "", cmap = "Spectral")
        ax.set_ylabel("Score")
        ax.set_xlabel(name)

    # Main functions
        
    def ensemble_classifier_plots(perfs, ad_data):
        fig, axs = plt.subplots(3, 3, figsize = (10, 10))
        axs = axs.flatten()
        if perfs is not None:
            scores_distro(axs[0], perfs)
            rocs(axs[1], perfs)    
            precrecs(axs[2], perfs)
            individual_performances(axs[3], perfs, "bedroc")
            individual_performances(axs[4], perfs, "auroc")
            individual_performances(axs[5], perfs, "aupr")
        if ad_data is not None:
            applicability(axs[6], perfs, ad_data, 0)
            applicability(axs[7], perfs, ad_data, 1)
            applicability(axs[8], perfs, ad_data, 2)
        plt.tight_layout()
        return fig, axs