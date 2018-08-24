'''
PLOTTING FUNCTIONS

It contains functions for all the plots

'''

from __future__ import division
import h5py
import math
import sys, os
#sys.path.append(os.path.join(sys.path[0], "../dbutils/"))
import Psql
#sys.path.append(os.path.join(sys.path[0], ".."))
import checkerUtils
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis 
import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from gensim import corpora, models, similarities
from scipy.stats import ks_2samp, fisher_exact
from scipy.spatial.distance import euclidean, cosine
from scipy.interpolate import splrep, splev
from numpy import matlib
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as mpl
import collections

random.seed(42)
np.random.seed(42)


# Mini functions

def get_plot_folder(table):
    if table not in checkerUtils.table_coordinates:
        if not os.path.exists("plots"): os.mkdir("plots")
        return "plots"
    coord = checkerUtils.table_coordinates[table]
    plot_folder = checkerUtils.coordinate2mosaic(coord) + "/" + CONFIG.default_plots_folder
    if not os.path.exists(plot_folder): os.mkdir(plot_folder)
    return plot_folder

def elbow(curve):
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def clustering_plot(Nc, A, B, C, table = None, plot_folder = None):
    
    if plot_folder is None:
        return

    if not table:
        color = "black"
    else:
        color = checkerUtils.table_color(table)

    sns.set_style("white")

    plt.figure(figsize=(4,4), dpi=600)

    fig = plt.subplot(111)
    plt.plot(Nc, A,  color = color, lw = 1, linestyle = ':')
    plt.plot(Nc, B,  color = color, lw = 1, linestyle = '--')
    plt.plot(Nc, C,  color = color, lw = 2, linestyle = '-')

    kidx = np.argmax(C)
    k = Nc[kidx]
    plt.plot([k,k], [0,1], color = color, lw = 1, linestyle = '-')

    plt.ylim(0,1)
    plt.xlim(np.min(Nc), np.max(Nc))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Cluster measures")

    plt.title("k: %d clusters" % k)

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    plt.savefig("%s/kmeans_kselect.png" % plot_folder)

    with open("%s/kmeans_kselect.tsv" % plot_folder, "w") as f:
        for i in xrange(len(Nc)):
            f.write("%d\t%f\t%f\t%f\n" % (Nc[i], A[i], B[i], C[i]))

    return k


def variance_plot(exp_var_ratios, table = None, variance_cutoff = 0.9, plot_folder = None):

    if plot_folder is None:
        return

    if not table:
        color = "black"
    else:
        color = checkerUtils.table_color(table)

    # Set the variance-explained cutoff.

    sns.set_style("white")

    cumsum = np.cumsum(exp_var_ratios)

    plt.figure(figsize=(4,4), dpi=600)
   
    fig = plt.subplot(111)
    x = [i for i in xrange(len(cumsum) + 1)]
    y = [0] + list(cumsum)
    plt.plot(x, y, color = color, lw = 2)

    # Varcut
    varcut = variance_cutoff
    for i in xrange(len(cumsum)):
        if cumsum[i] > varcut:
            break
    i90 = i
    plt.plot([0., i90+1], [varcut, varcut], color = color, linestyle = "-")
    plt.plot([i90+1, i90+1], [0, varcut], color = color, linestyle = "-")
    plt.scatter([i90+1], [varcut], color = "white", edgecolor = color, lw=1.5, zorder = 3, s = 50)

    # Elbow point
    curve = cumsum[:i90+1]
    ielb = elbow(curve)
    elb = cumsum[ielb]
    plt.plot([0., ielb+1], [elb, elb], color = color, linestyle = "--")
    plt.plot([ielb+1, ielb+1], [0, elb], color = color, linestyle = "--")
    plt.scatter([ielb+1], [elb], color = "white", edgecolor = color, lw=1.5, zorder = 3, s = 50)

    plt.grid(linestyle="-.", color = color, lw = 0.3)
    plt.ylim(0,1)
    plt.xlim(0,len(cumsum))
    plt.xlabel("Latent variables")
    plt.ylabel("Proportion of variance explained")

    plt.title("%.1f: %d, elbow: %d" % (varcut, i90 + 1, ielb + 1))

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    plt.savefig("%s/variance_explained.png" % plot_folder)
    
    with open("%s/variance_explained.tsv" % plot_folder, "w") as f:
        for i in xrange(len(x)):
            f.write("%f\t%f\n" % (x[i], y[i]))

    return i90, ielb


# Background distribution of euclidean distances

def euclidean_background(inchikey_vec, inchikeys = None, B = 100000, metric = euclidean):
   
    # Check if it is a numpy array
 
    if type(inchikey_vec).__module__ == np.__name__:
        idxs = [i for i in xrange(inchikey_vec.shape[0])]
        bg = []
        for _ in xrange(B):
            i, j = random.sample(idxs, 2)
            bg += [metric(inchikey_vec[i,:], inchikey_vec[j,:])]

    else:

        if inchikeys is None:
            inchikeys = np.array([k for k,v in inchikey_vec.iteritems()])

        bg = []
        for _ in xrange(B):
            ik1, ik2 = random.sample(inchikeys, 2)
            bg += [metric(inchikey_vec[ik1], inchikey_vec[ik2])]

    i = 0
    PVALS = [(0, 0., i)] # DISTANCE, RANK, INTEGER
    i += 1
    percs = [0.001, 0.01, 0.1] + list(np.arange(1,100))
    for perc in percs:
        PVALS += [(np.percentile(bg, perc), perc/100., i)]
        i += 1
    PVALS += [(np.max(bg), 1., i)]

    return PVALS


# Validate using moa and KS test

def for_the_validation(inchikey_dict, prefix, file_folder = None):

    if file_folder  is None:
        f = open(os.path.dirname(os.path.abspath(__file__))+"/data/%s_validation.tsv" % prefix, "r")
    else:
        f = open(file_folder+"/%s_validation.tsv" % prefix, "r")
    S = set()
    D = set()
    inchikeys = set()
    for l in f:
        l = l.rstrip("\n").split("\t")
        inchikeys.update([l[0], l[1]])
        if int(l[2]) == 1:
            S.update([(l[0], l[1])])
        else:
            if len(D) < 100000:
                D.update([(l[0], l[1])])
            else:
                pass
    f.close()

    d = {}
    for inchikey in inchikeys:
        try:
            d[inchikey] = inchikey_dict[inchikey]
        except:
            continue
    inchikeys = inchikeys.intersection(d.keys())
    S = set([x for x in S if x[0] in inchikeys and x[1] in inchikeys])
    D = set([x for x in D if x[0] in inchikeys and x[1] in inchikeys])
    d = dict((k, d[k]) for k in inchikeys)

    return S, D, d


def label_validation(inchikey_lab, label_type, table = None, prefix = "moa", plot_folder = None, files_folder = None):
    
    if plot_folder is None:
        return

    if not table:
        color = "black"
    else:
        color = checkerUtils.table_color(table)

    S, D, d = for_the_validation(inchikey_lab, prefix, files_folder)

    yy, yn, ny, nn = 0, 0, 0, 0
    
    for k in S:
        if d[k[0]] == d[k[1]]:
            yy += 1
        else:
            yn += 1
    for k in D:
        if d[k[0]] == d[k[1]]:
            ny += 1
        else:
            nn += 1

    M = np.array([[yy, yn],[ny, nn]])

    odds, pval = fisher_exact(M, alternative = "greater")

    sns.set_style("white")
    plt.figure(figsize=(4,4), dpi=600)
    fig = plt.subplot(111)
    plt.bar([1,2], [M[0,0] / (M[1,0] + M[0,0]) * 100, M[0,1] / (M[1,1] + M[0,1]) * 100], color = [color, "white"], edgecolor = color, lw = 2)
    plt.xticks([1,2], ["Same", "Different"])
    for h in np.arange(10,100,10):
        plt.plot([-1,3], [h,h], linestyle = '--', color = color, lw = 0.3)
    plt.ylim((0, 100))
    plt.xlim((0.5, 2.5))
    plt.ylabel("% in same cluster")

    plt.title("Odds: %.2f, P-val: %.2g" % (odds, pval))

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    plt.tight_layout()
    plt.savefig("%s/%s_%s_ft_validation.png" % (plot_folder, prefix, label_type))

    with open("%s/%s_%s_ft_validation.tsv" % (plot_folder, prefix, label_type), "w") as f:
        S  = "%s\t%d\n" % ("yy", yy)
        S += "%s\t%d\n" % ("yn", yn)
        S += "%s\t%d\n" % ("ny", ny)
        S += "%s\t%d\n" % ("nn", nn)
        S += "odds\t%.2f\n" % (odds)
        S += "pval\t%.2g\n" % (pval)
        f.write(S)

    return odds, pval


def vector_validation(inchikey_vec, vector_type, table = None, prefix = "moa", plot_folder = None, files_folder = None, distance = "euclidean"):

    if plot_folder is None:
        return

    if not table:
        color = "black"
    else:
        color = checkerUtils.table_color(table)

    S, D, d = for_the_validation(inchikey_vec, prefix,files_folder)

    if distance == "euclidean":
        distance_metric = euclidean
    elif distance == "cosine":
        distance_metric = cosine
    else:
        sys.exit("Unrecognized distance %s" % distance)
    
    S = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in S]))
    D = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in D]))
        
    ks = ks_2samp(S, D)

    N = len(d)
 
    d = None
 
    # Euclidean distance plot
 
    sns.set_style("white")

    plt.figure(figsize=(4,4), dpi=600)
    fig = plt.subplot(111)

    cS = np.cumsum(S)
    cS = cS / np.max(cS)
    
    cD = np.cumsum(D)
    cD = cD / np.max(cD)

    plt.plot(D, cD, color = color, linestyle = "--")
    plt.plot(S, cS, color = color, linestyle = "-", lw = 2)
    
    plt.grid(linestyle="-.", color = color, lw = 0.3)
    
    plt.ylim(0,1)
    plt.xlim(np.min([np.min(S), np.min(D)]) ,np.max([np.max(S), np.max(D)]))
    plt.xlabel("Euclidean distance")
    plt.ylabel("Cumulative proportion")

    plt.title("D: %.2f, P-val: %.2g, N: %d" % (ks[0], ks[1], N))

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    plt.savefig("%s/%s_%s_ks_validation.png" % (plot_folder, prefix, vector_type))

    with open("%s/%s_%s_ks_validation_D.tsv" % (plot_folder, prefix, vector_type), "w") as f:
        for i in xrange(len(D)):
            f.write("%f\t%f\n" % (D[i], cD[i]))    

    with open("%s/%s_%s_ks_validation_S.tsv" % (plot_folder, prefix, vector_type), "w") as f:
        for i in xrange(len(S)):
            f.write("%f\t%f\n" % (S[i], cS[i]))

    # ROC curve

    plt.figure(figsize=(4,4), dpi=600)
    fig = plt.subplot(111)

    Scores = np.concatenate((-S, -D))
    Truth = np.array(len(S)*[1] + len(D)*[0])

    auc_score = roc_auc_score(Truth, Scores)

    fpr, tpr, thr = roc_curve(Truth, Scores)

    plt.plot([0,1], [0,1], color = color, linestyle = "--")
    plt.plot(fpr, tpr, color = color, linestyle = "-", lw = 2)

    plt.grid(linestyle="-.", color = color, lw = 0.3)

    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.title("AUC: %.2f, N: %d" % (auc_score, N))

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    plt.savefig("%s/%s_%s_auc_validation.png" % (plot_folder, prefix, vector_type))

    with open("%s/%s_%s_auc_validation.tsv" % (plot_folder, prefix, vector_type), "w") as f:
        for i in xrange(len(fpr)):
            f.write("%f\t%f\n" % (fpr[i], tpr[i]))

    return ks, auc_score


# Matrix plot

def matrix_plot(table, sig_folder, plot_folder = None):

    if plot_folder is None:
        return

    sns.set_style("white")

    color = checkerUtils.table_color(table)
    with h5py.File(sig_folder + "/sig.h5") as hf:
        Mols = len(hf["keys"])
        Vars = len(hf["V"][0])

    plt.figure(figsize = (4,4), dpi=300)

    fig = plt.subplot(111)

    ax = plt.gca()

    xmax = 3
    ymax = 6

    plt.xlim(0,xmax)
    plt.ylim(0,ymax)

    for v1 in range(1, xmax):
        plt.plot([v1, v1], [0, ymax], "-.", lw=0.5, color = color)
    for h1 in range(1, ymax):
        plt.plot([0, xmax], [h1, h1], "-.", lw=0.5, color = color)

    ax.add_patch(patches.Rectangle((0,0), math.log10(Vars), math.log10(Mols), color = color))

    plt.yticks([t for t in xrange(7)])
    plt.xticks([t for t in xrange(4)])

    ax.set_xlabel("Latent variables (log10)")
    ax.set_ylabel("Molecules (log10)")

    fig.axes.spines["bottom"].set_color(color)
    fig.axes.spines["top"].set_color(color)
    fig.axes.spines["right"].set_color(color)
    fig.axes.spines["left"].set_color(color)

    fig.patch.set_facecolor("white")

    plt.savefig("%s/matrix_plot.png" % plot_folder)


# Projection plot

def projection_plot(table, Proj, bw = None, levels = 5, dev = None, s = None, transparency = 0.5, plot_folder = None):

    if plot_folder is None:
        return

    color = checkerUtils.table_color(table)
    gray  = checkerUtils.gray

    if dev:
        noise_x = np.random.normal(0, dev, Proj.shape[0])
        noise_y = np.random.normal(0, dev, Proj.shape[0])

    if not s:
        s = np.max([0.3, -4e-6*Proj.shape[0] + 4])

    fig = plt.figure(figsize = (5,5), dpi = 600)
    ax = fig.add_subplot(111)

    ax.set_facecolor(color)
    if dev:
        X = Proj[:,0] + noise_x
        Y = Proj[:,1] + noise_y
    else:
        X = Proj[:,0]
        Y = Proj[:,1]
    ax.scatter(X, Y, alpha = transparency, s = s, color = "white")
    ax.grid(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.savefig("%s/largevis_scatter.png" % plot_folder)

    def make_cmap(colors, position=None, bit=False):
        bit_rgb = np.linspace(0,1,256)
        if position == None:
            position = np.linspace(0,1,len(colors))
        else:
            if len(position) != len(colors):
                sys.exit("position length must be the same as colors")
            elif position[0] != 0 or position[-1] != 1:
                sys.exit("position must start with 0 and end with 1")
        if bit:
            for i in range(len(colors)):
                colors[i] = (bit_rgb[colors[i][0]],
                             bit_rgb[colors[i][1]],
                             bit_rgb[colors[i][2]])
        cdict = {'red':[], 'green':[], 'blue':[]}
        for pos, color in zip(position, colors):
            cdict['red'].append((pos, color[0], color[0]))
            cdict['green'].append((pos, color[1], color[1]))
            cdict['blue'].append((pos, color[2], color[2]))

        cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
        return cmap

    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i+int(lv/3)], 16) for i in range(0, lv, int(lv/3)))

    def get_cmap(color):
        colors = [hex_to_rgb(c) for c in [color, gray]]
        return make_cmap(colors, bit = True)

    sns.set_style("white")

    cmap = get_cmap(color)

    xmin, xmax = np.min(Proj[:,0]), np.max(Proj[:,0])
    ymin, ymax = np.min(Proj[:,1]), np.max(Proj[:,1])

    margin = 0.1

    xran = xmax - xmin
    xmin = xmin - margin*xran
    xmax = xmax + margin*xran
    yran = ymax - ymin
    ymin = ymin - margin*yran
    ymax = ymax + margin*yran
    
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    Proj = np.vstack({tuple(row) for row in Proj})
    values = np.vstack([Proj[:,0], Proj[:,1]])
    kernel = gaussian_kde(values, bw_method = bw)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot
    fig = plt.figure(figsize=(5,5), dpi = 600)
    
    ax = fig.add_subplot(111)
    
    cut = np.percentile(f, 95)

    f[f > cut] = cut
    levels = np.linspace(0, cut, num = levels)
    f[f == np.min(f)] = 0

    ax.grid(False)
    plt.contourf(xx, yy, f, list(levels) + [levels[-1] + 5*(levels[-1] - levels[-2])], cmap = cmap)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    ax.axes.spines["bottom"].set_color(color)
    ax.axes.spines["top"].set_color(color)
    ax.axes.spines["right"].set_color(color)
    ax.axes.spines["left"].set_color(color)

    plt.savefig("%s/largevis.png" % plot_folder, bbox_inches = "tight", pad_inches = 0)

    xlim = ax.axes.get_xlim()
    ylim = ax.axes.get_ylim()

    return xlim, ylim

