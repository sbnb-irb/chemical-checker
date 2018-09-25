#!/miniconda/bin/python

# Canonical correlation analysis.

# Imports

from __future__ import division
import sys, os
import h5py
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,coordinate2mosaic
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
import random
random.seed(42)

# Variables
log = None

def batcher(X, size):
    for pos in xrange(0, X.shape[0], size):
        yield (pos, pos+size)

def z_transform(rho, n):
    return (np.sqrt(n - 3) / 2.)*np.log((1 + rho) / (1 - rho))

def pls_single(X,Y):
    r1, r2 = [], []
    for froto in batcher(X, 100000):
        x = X[froto[0]:froto[1]]
        y = Y[froto[0]:froto[1]]
        plsca = PLSRegression(n_components=2, max_iter = 1000)
        plsca.fit(x,y)
        X_r, Y_r = plsca.transform(x,y)
        r1 += [pearsonr(X_r[:,0], Y_r[:,0])[0]]
        r2 += [pearsonr(X_r[:,1], Y_r[:,1])[0]]
    return np.median(r1), np.median(r2)


def pls(X, Y, B = 100):
    r1, r2 = pls_single(X, Y)
    r1rand, r2rand = [], []
    Xr = X.copy()
    for _ in xrange(B):
        np.random.shuffle(Xr)
        r1r, r2r = pls_single(Xr,Y)
        r1rand += [r1r]
        r2rand += [r2r]
    p1 = len([1 for r in r1rand if r > r1]) / float(B)
    p2 = len([1 for r in r2rand if r > r2]) / float(B)
    z1 = z_transform(r1, X.shape[0])
    z2 = z_transform(r2, X.shape[0])
    return r1, z1, p1, r1/np.mean(r1rand), r2, z2, p2, r2/np.mean(r2rand)


def cross_validate(X, Y, B = 100):
    r1, r2, z1, z2, p1, p2 = [], [], [], [], [], []
    # Random data
    for _ in xrange(100):
        idxs   = random.sample([i for i in xrange(X.shape[0])], np.min([X.shape[0], 100000]))
        n_test = int(np.max([5, len(idxs) / 10]))
        test   = idxs[:n_test]
        train  = idxs[n_test:]
        X_train = X[train,:len(train)]
        Y_train = Y[train,:len(train)]
        X_test = X[test,:len(train)]
        Y_test = Y[test,:len(train)]
        plsca = PLSRegression(n_components=2, max_iter = 1000)
        plsca.fit(X_train, Y_train)
        X_r, Y_r = plsca.transform(X_test, Y_test)
        cor1 = pearsonr(X_r[:,0], Y_r[:,0])
        cor2 = pearsonr(X_r[:,1], Y_r[:,1])
        r1 += [cor1[0]]
        r2 += [cor2[0]]
        z1 += [z_transform(cor1[0], X_r.shape[0])]
        z2 += [z_transform(cor2[0], X_r.shape[0])]
        p1 += [cor1[1]]
        p2 += [cor2[1]]
    return np.median(r1), np.median(z1), np.median(p1), np.median(r2), np.median(z2), np.median(p2)

if __name__ == '__main__':

    
    coordinate_1, coordinate_2,dbname = sorted(sys.argv[1].split("---"))
    

    max_comp = 50
    
    # Get molecules
    
    with h5py.File(coordinate2mosaic(coordinate_1)+ "sig.h5") as hf:
        inchikeys1 = hf['keys'][:]
        n1 = len(inchikeys1)
    
    with h5py.File(coordinate2mosaic(coordinate_2)+ "sig.h5") as hf:
        inchikeys2 = hf['keys'][:]
        n2 = len(inchikeys2)
    
    inchikeys  = set(inchikeys1).intersection(set(inchikeys2))
    inchikeys1 = dict((inchikeys1[i], i) for i in xrange(len(inchikeys1)))
    inchikeys2 = dict((inchikeys2[i], i) for i in xrange(len(inchikeys2)))
    
    if len(inchikeys) < 10: 
        print "Analysis not done! Only %d molecules in common between %s and %s..." % (len(inchikeys), coordinate_1, coordinate_2)
        return
    
    print len(inchikeys)
    
    # Canonical correlation analysis
    
    print "Doing canonical correlation analysis"


    with h5py.File(coordinate2mosaic(coordinate_1)+ "sig.h5") as hf:
        X = hf["V"][:,:max_comp]
        X = np.array([X[inchikeys1[ik],:] for ik in inchikeys])
        X = X[:,:X.shape[0]]
        inchikeys1 = {}
        comps_1 = X.shape[1]
    
    with h5py.File(coordinate2mosaic(coordinate_2)+ "sig.h5") as hf:
        Y = hf["V"][:,:max_comp]
        Y = np.array([Y[inchikeys2[ik],:] for ik in inchikeys])
        Y = Y[:,:Y.shape[0]]
        inchikeys2 = {}
        comps_2 = Y.shape[1]
    
    
    r1, z1, p1, o1, r2, z2, p2, o2           = pls(X,Y)
    
    cv_r1, cv_z1, cv_p1, cv_r2, cv_z2, cv_p2 = cross_validate(X, Y) 
    
    print "Inserting to database"
    
    Psql.query("DELETE FROM coordinate_correlation WHERE coord_a = '%s' and coord_b = '%s'" % (coordinate_1, coordinate_2), dbname) 
    Psql.query("INSERT INTO coordinate_correlation VALUES ('%s', '%s', %d, %d, %d, %.2f, %.2f, %.2g, %.1f, %.2f, %.2f, %.2g, %.1f, %.2f, %.2f, %.2g, %.2f, %.2f, %.2g)" % (coordinate_1, coordinate_2, n1, n2, len(inchikeys), r1, z1, p1, o1, r2, z2, p2, o2, cv_r1, cv_z1, cv_p1, cv_r2, cv_z2, cv_p2), dbname)
