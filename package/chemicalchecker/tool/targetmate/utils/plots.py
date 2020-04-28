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
        perfs = self.valid["test"]["perfs"]
        x = []
        y = []
        for i, k in enumerate(mets):
            x += [i]
            y += [perfs[k][0][j]]
        ax.scatter(x, y, color = "grey")

    def _ranking_metrics(self, ax, label):
        self.__log.debug("Ranking metrics plot")
        j = self._get_j(label)
        mets = ["auroc", "aupr", "bedroc"]
        perfs = self.valid["test"]["perfs"]
        y = []
        for spl in range(0, self.valid["n_splits"]):
            x = []
            y_ = []
            for i, k in enumerate(mets):
                y_ += [perfs[k][spl][j]]
                x += [i]
            y += [y]
        print(x, y)
        y = np.array(y)
        y = np.mean(y, axis = 0)
        ax.scatter(x, y, color = "grey")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in mets])
        ax.set_ylabel("Performance")
        ax.set_title("Ranking metrics")
            
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
        y_pred   = self.valid["test"]["y_pred"][0][:,j][self.valid["test"]["y_true"][0] == j]
        y_pred_  = self.valid["test"]["y_pred"][0][:,j][self.valid["test"]["y_true"][0] != j]
        x = np.arange(0, 1, 0.01)
        density  = stats.kde.gaussian_kde(y_pred)
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
    
    def roc_curve(self, ax, label):
        self.__log.debug("ROC curve")
        j = self._get_j(label)
        y_true_tr = np.abs(self.valid["train"]["y_true"][0] - (1-j))
        y_pred_tr = self.valid["train"]["y_pred"][0][:,j]
        fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
        auc_tr = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = "Train", color = coord_color("C"))
        y_true_tr = np.abs(self.valid["test"]["y_true"][0] - (1-j))
        y_pred_tr = self.valid["test"]["y_pred"][0][:,j]
        fpr, tpr, _ = roc_curve(y_true_tr, y_pred_tr)
        auc_ts = auc(fpr, tpr)
        ax.plot(fpr, tpr, label = "Test", color = coord_color("A"))
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC %.2f / %.2f" % (auc_tr, auc_ts))
        ax.legend()

    def pr_curve(self, ax, label):
        self.__log.debug("PR curve")
        j = self._get_j(label)
        y_true_tr = np.abs(self.valid["train"]["y_true"][0] - (1-j))
        y_pred_tr = self.valid["train"]["y_pred"][0][:,j]
        pre, rec, _ = precision_recall_curve(y_true_tr, y_pred_tr)
        auc_tr = auc(rec, pre)
        ax.plot(rec, pre, label = "Train", color = coord_color("C"))
        y_true_ts = np.abs(self.valid["test"]["y_true"][0] - (1-j))
        y_pred_ts = self.valid["test"]["y_pred"][0][:,j]
        pre, rec, _ = precision_recall_curve(y_true_ts, y_pred_ts)
        auc_ts = auc(rec, pre)
        ax.plot(rec, pre, label = "Test", color = coord_color("A"))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR %.2f / %.2f" % (auc_tr, auc_ts))
        ax.legend()
        
    def confidence_level(self, ax):
        self.__log.debug("Confidence level plot")
        y_pred = self.valid["test"]["y_pred"][0]
        y_true = self.valid["test"]["y_true"][0]
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
        
    def validity(self, ax):
        self.__log.debug("Validity plot")
        y_pred = self.valid["test"]["y_pred"][0]
        y_true = self.valid["test"]["y_true"][0]
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
        
    def efficiency(self, ax):
        self.__log.debug("Efficiency plot")
        """Fraction of single-class predictions that are correct."""
        y_pred = self.valid["test"]["y_pred"][0]
        y_true = self.valid["test"]["y_true"][0]
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
        ax.set_ylabel("Correct predictions")
        ax.set_title("Efficiency")
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
               
    def both_pvalues(self, ax):
        self.__log.debug("Both p-values plot")
        y_pred = self.valid["test"]["y_pred"][0]
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
        
    def calibration(self, ax, label):
        self.__log.debug("Calibration plot")
        if label == "Active":
            color = coord_color("C")
        else:
            color = coord_color("A")
        j = self._get_j(label)
        y_pred = self.valid["test"]["y_pred"][0][:,j][self.valid["test"]["y_true"][0] == j]
        ax.plot([(i+1) for i in range(0, len(y_pred))], sorted(y_pred), color = color)
        ax.set_title("Calibr. %s" % label)
        ax.set_xlabel("Test set compounds")
        ax.set_ylabel("PS")
        ax.set_ylim(0,1)
                        
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

