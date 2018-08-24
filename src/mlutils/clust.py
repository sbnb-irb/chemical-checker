#!/miniconda/bin/python


# Imports

from __future__ import division
from csvsort import csvsort
import csv
import h5py
import json
import uuid
import pqkmeans
import sys, os, glob
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
from checkerUtils import logSystem,log_data,tqdm_local
import Psql
import numpy as np
import random
import subprocess
import shelve
from auto_plots import label_validation, clustering_plot, euclidean_background
import collections
import argparse
import bisect
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import rankdata

# Variables

random.seed(42)

# Models files

clustencoder_file     = "clustencoder.h5"
clustcentroids_file   = "clustcentroids.h5"
clust_info_file       = "clust_stats.json"
clust_output          = 'clust.h5'
bg_pq_euclideans_file = "bg_pq_euclideans.h5"




# Functions

def get_A(encoder):
    A = np.zeros((encoder.M, encoder.Ks, encoder.Ks))
    for m in xrange(encoder.M):
        A[m] = squareform(pdist(encoder.codewords[m], 'sqeuclidean'))
    return A


# Functions

def symmetric_distance(x,y):
    d = 0
    for m in xrange(A.shape[0]):
        d += A[m][x[m],y[m]]
    return d

def smooth(x,max_k, window_len=None, window='hanning'):
    if window_len is None:
        window_len = int(max_k / 10) + 1
    if window_len % 2 == 0:
        window_len += 1
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    n = int((window_len - 1) / 2)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[n:-n]

def inertia(V_pqcode, labels, centroids):
    ines = 0
    for i in xrange(V_pqcode.shape[0]):
        ines += symmetric_distance(V_pqcode[i], centroids[labels[i]])
    return ines

def dispersion(centroids, sig_dist):
    if len(centroids) == 1: return None
    return np.sum(pdist(centroids, metric = symmetric_distance) < sig_dist)

def monotonize(v, up = True):
    if up: return np.mean(np.array([np.maximum.accumulate(v), np.minimum.accumulate(v[::-1])[::-1]]), axis = 0)
    else:  return np.mean(np.array([np.minimum.accumulate(v), np.maximum.accumulate(v[::-1])[::-1]]), axis = 0) 

def minmaxscaler(v):
    v = np.array(v)
    Min = np.min(v)
    Max = np.max(v)
    return (v - Min) / (Max - Min)        


def get_balance(V_pqcode, centroids, labels, balance, tmp,log):

    if balance is None: return labels

    if balance < 1: 
        print "Balance is smaller than 1. I don't understand. Anyway, I just don't balance."
        return labels
    
    S = np.ceil((V_pqcode.shape[0] / k)*balance)

    clusts = [None] * V_pqcode.shape[0]
    counts = [0] * k

    tmpfile = tmp + "_dists.csv"

    with open(tmpfile, "w") as f:

        for i, v in tqdm_local(log,enumerate(V_pqcode)):
            for j, c in enumerate(centroids):
                d = symmetric_distance(c, v)
                f.write("%d,%d,%010d\n" % (i,j,d))
    
    csvsort(tmpfile, [2], has_header = False)

    with open(tmpfile, "r") as f:
        for r in tqdm_local(log,csv.reader(f)):
            item_id = int(r[0])
            cluster_id = int(r[1])
            if counts[cluster_id] >= S:
                continue
            if clusts[item_id] is None:
                clusts[item_id] = cluster_id
                counts[cluster_id] += 1

    os.remove(tmpfile)

    return clusts 

def clustering( table,filename = None,outfile = None,models_folder = None,plots_folder = None,max_k = None, 
                       min_k = 1, n_points = 100,k = None, num_subdim = 8, Ks = 256, 
                        recycle = False,significance = 0.05, B_euclidean= 1000000, balance = None, log = None, tmpDir = ''):
    

    checkercfg = checkerconfig.checkerConf()
    
    if tmpDir != '':
        tmp =  os.path.join(tmpDir,str(uuid.uuid4()))
    else:
        tmp = str(uuid.uuid4())
        
    if outfile is None:
        outfile = checkercfg.coordinate2mosaic(coord) + "/" + clust_output
        
    if models_folder is None:
        models_folder = checkercfg.coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_MODELS_FOLDER
    
    if plots_folder is None:
        plots_folder = checkercfg.coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_PLOTS_FOLDER
        
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)
    if not os.path.exists(models_folder): os.makedirs(models_folder)
    # Encode matrix...
    
    if not recycle:
        encoder = pqkmeans.encoder.PQEncoder(num_subdim=num_subdim, Ks=Ks)
        inds = [i for i in xrange(int(int(V.shape[1]/num_subdim)*num_subdim))]
        random.shuffle(inds)
        inds = np.array(inds).astype(np.int)
        V = V[:,inds]
        encoder.fit(V)
        A = get_A(encoder)
        with h5py.File(os.path.join(models_folder,clustencoder_file), "w") as hf:
            hf.create_dataset("codewords", data = encoder.codewords)
            hf.create_dataset("A", data = A)
            hf.create_dataset("inds", data = inds)
    else:
        encoder = pqkmeans.encoder.PQEncoder(num_subdim=num_subdim, Ks=Ks)
        with h5py.File(os.path.join(models_folder,clustencoder_file), "r") as hf:
            trained_encoder = pqkmeans.encoder.TrainedPQEncoder(hf["codewords"][:], encoder.code_dtype)
            A = hf["A"][:]
            inds = hf["inds"][:]
        encoder.trained_encoder = trained_encoder
        encoder.Ds = trained_encoder.Ds
        V = V[:,inds]
    
    V_pqcode = encoder.transform(V)
    
    del V
    
    # Load matrix...
    
    with h5py.File(filename, "r") as hf:
        inchikeys = hf["keys"][:]
        V = hf["V"][:]
        try:
            with open(filename.replace(".h5", "_stats.json"), "r") as f:
                d = json.load(f)
                Vn, Vm = d['molecules'], d['elbow_variables']
        except:
            log_data(log, "No elbow known, taking shape of input matrix (divided by 2)")
            Vn, Vm = V.shape[0], V.shape[1] / 2
    
    if V.shape[1] < num_subdim:
        V = np.hstack((V, np.zeros((V.shape[0], num_subdim - V.shape[1]))))
    
    # Do the clustering
    
    if recycle and not k:
    
        k = json.load(open(clust_info_file, "r"))["k"]
    
    elif not k:
    
        # Do reference distributions for the gap statistic
    
        if not min_k:
            min_k = 1
        if not max_k:
            max_k = int(np.sqrt(V_pqcode.shape[0]))
    
        # Doing euclidean background
        
        pvals = np.array(euclidean_background(V_pqcode, metric = symmetric_distance, B = B_euclidean))
        sig_dist = pvals[bisect.bisect_left(pvals[:,1], significance), 0]
    
        with h5py.File(os.path.join(models_folder,bg_pq_euclideans_file), "w") as hf:
            hf.create_dataset("integer" , data = pvals[:,2])
            hf.create_dataset("pvalue"  , data = pvals[:,1])
            hf.create_dataset("distance", data = pvals[:,0])
    
        # Doing the clustering        
    
        cluster_range = np.arange(min_k, max_k, step = np.max([int((max_k - min_k) / n_points), 1]))
        inertias = []
        disps    = []
        for k in tqdm_local(log,cluster_range):
            km = pqkmeans.clustering.PQKMeans(encoder = encoder, k = k)
            km.fit(V_pqcode)
            centroids = km.cluster_centers_
            labels = rankdata(km.labels_, method = "dense") - 1
            inertias += [inertia(V_pqcode, labels, centroids)]
            disps    += [dispersion(centroids, sig_dist)]
        disps[0] = disps[1]
    
        # Smooting, monotonizing, and combining the scores
    
        Ncs = np.arange(min_k, max_k)
        D   = minmaxscaler(np.interp(Ncs, cluster_range, smooth(monotonize(np.array(disps), True) , max_k)))
        I   = minmaxscaler(np.interp(Ncs, cluster_range, smooth(monotonize(np.array(inertias), False),  max_k)))
    
        alpha = Vm / (Vm + np.sqrt(Vn / 2.))
    
        S = np.abs((I**(1 - alpha)) - (D**(alpha)))
        S = minmaxscaler(-smooth(S,  max_k))
    
        k = clustering_plot(Ncs, I, D, S, table = table)
    
    
    
    
    if not recycle:
    
        log_data(log, "Clustering with k = %d" % k)
    
        km = pqkmeans.clustering.PQKMeans(encoder = encoder, k = k)
        km.fit(V_pqcode)
    
        centroids = np.array(km.cluster_centers_)
    
        with h5py.File(os.path.join(models_folder,clustcentroids_file), "w") as hf:
            hf.create_dataset("centroids", data = centroids)    
    
        centroids = centroids.astype(encoder.code_dtype)
    
    else:
    
        log_data(log, "Loading learned clustering with k = %d" % k    )
        
        with h5py.File(os.path.join(models_folder,clustcentroids_file), "r") as hf:
            centroids = hf["centroids"][:].astype(encoder.code_dtype)
        
    
    log_data(log, "Predicting...")
    
    labels = cdist(V_pqcode, centroids, metric = symmetric_distance).argmin(axis = 1)
    
    if not recycle:
        log_data(log, "Balancing...")
        labels = get_balance(V_pqcode, centroids, labels, balance, tmp,log)
    
    log_data(log, "Saving matrix...")
    
    with h5py.File(outfile, "w") as hf:
        hf.create_dataset("labels", data = labels)
        hf.create_dataset("keys", data = inchikeys)
        hf.create_dataset("V_pqcode", data = V_pqcode)
    
    if recycle:
        sys.exit("Done")
    
    # MOA validation
    
    log_data(log, "Doing validations")
    
    inchikey_clust = shelve.open(tmp+".dict", "n")
    for i in tqdm_local(log,xrange(len(inchikeys))):
        inchikey_clust[inchikeys[i]] = labels[i]
    odds_moa, pval_moa = label_validation(inchikey_clust, "clust", table, prefix = "moa", plot_folder = plots_folder,files_folder = tmpDir)
    odds_atc, pval_atc = label_validation(inchikey_clust, "clust", table, prefix = "atc", plot_folder = plots_folder,files_folder = tmpDir)
    inchikey_clust.close()
    
    
    log_data(log, "Cleaning")
    for filename in glob.glob(tmp+".dict*") :
        os.remove(filename)
    
    # Only save if we've done the screening of k's. 
    if not k:
        log_data(log, "Saving info")
        INFO = {
        "k": k,
        "odds_moa": odds_moa,
        "pval_moa": pval_moa,
        "odds_atc": odds_atc,
        "pval_atc": pval_atc 
        }   
        with open(checkercfg.coordinate2mosaic(coord) + "/"+clust_info_file, 'w') as fp:
            json.dump(INFO, fp) 
            
        
if __name__ == '__main__':


    # Parse arguments

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--table', default = None, type=str, help = 'Table')
    parser.add_argument('--filename', default = 'sig.h5', type=str, help = 'Signature file.')
    parser.add_argument('--outfile', default = 'clust.h5', type=str, help = 'Output file.')
    parser.add_argument('--models_folder', default = None, type = str, help = 'Models folder')
    parser.add_argument('--plots_folder', default = None, type = str, help = 'Plots folder')
    parser.add_argument('--max_k', default = None, type=int, help = 'Maximum number of clusters')
    parser.add_argument('--min_k', default = 1, type=int, help = 'Minimum number of clusters')
    parser.add_argument('--n_points', default = 100, type=int, help = 'Number of points to calculate')
    parser.add_argument('--k', default = None, type = int, help = 'Exact number of clusters to do computation')
    parser.add_argument('--num_subdim', default = 8, type = int, help = 'Splitting of the PQ encoder')
    parser.add_argument('--Ks', default = 256, type = int, help = 'Bits of the PQ encoder')
    parser.add_argument('--recycle', default = False, action = 'store_true', help = 'Recycle stored models')
    parser.add_argument('--significance', default = 0.05, type = float, help = 'Distance significance cutoff')
    parser.add_argument('--B_euclidean', default = 1000000, type = int, help = 'Number of samples in the background euclideans')
    parser.add_argument('--balance', default = None, type = float, help = 'If 1, all clusters are of equal size. Greater values are increasingly more imbalanced')
    
    args = parser.parse_args()
    
    clustering( args.table,args.filename,args.outfile,args.models_folder,arg.plots_folder,args.max_k,args.min_k,
                        args.n_points,args.k,args.num_subdim,args.Ks,args.recycle,args.significance,args.B_euclidean, args.balance,None)

