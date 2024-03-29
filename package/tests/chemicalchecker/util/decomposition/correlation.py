"""Canonical correlation analysis."""
import random
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression


def batcher(X, size):
    for pos in range(0, X.shape[0], size):
        yield (pos, pos + size)


def z_transform(rho, n):
    try:
        return (np.sqrt(n - 3) / 2.) * np.log((1 + rho) / (1 - rho))
    except Exception:
        print('WARN divition by 0')
        return 0


def pls_single(X, Y):
    r1, r2 = [], []
    for froto in batcher(X, 100000):
        x = X[froto[0]:froto[1]]
        y = Y[froto[0]:froto[1]]
        plsca = PLSRegression(n_components=2, max_iter=1000)
        plsca.fit(x, y)
        X_r, Y_r = plsca.transform(x, y)
        r1 += [pearsonr(X_r[:, 0], Y_r[:, 0])[0]]
        r2 += [pearsonr(X_r[:, 1], Y_r[:, 1])[0]]
    return np.median(r1), np.median(r2)


def pls(X, Y, B=100):
    r1, r2 = pls_single(X, Y)
    r1rand, r2rand = [], []
    Xr = X.copy()
    for _ in tqdm(range(B)):
        np.random.shuffle(Xr)
        r1r, r2r = pls_single(Xr, Y)
        r1rand += [r1r]
        r2rand += [r2r]
    p1 = len([1 for r in r1rand if r > r1]) / float(B)
    p2 = len([1 for r in r2rand if r > r2]) / float(B)
    z1 = z_transform(r1, X.shape[0])
    z2 = z_transform(r2, X.shape[0])
    return r1, z1, p1, r1 / np.mean(r1rand), r2, z2, p2, r2 / np.mean(r2rand)


def cross_validate(X, Y, B=100):
    r1, r2, z1, z2, p1, p2 = [], [], [], [], [], []
    # Random data
    for _ in tqdm(range(100)):
        idxs = random.sample([i for i in range(X.shape[0])],
                             np.min([X.shape[0], 100000]))
        n_test = int(np.max([5, len(idxs) / 10]))
        test = idxs[:n_test]
        train = idxs[n_test:]
        X_train = X[train, :len(train)]
        Y_train = Y[train, :len(train)]
        X_test = X[test, :len(train)]
        Y_test = Y[test, :len(train)]
        plsca = PLSRegression(n_components=2, max_iter=1000)
        plsca.fit(X_train, Y_train)
        X_r, Y_r = plsca.transform(X_test, Y_test)
        cor1 = pearsonr(X_r[:, 0], Y_r[:, 0])
        cor2 = pearsonr(X_r[:, 1], Y_r[:, 1])
        r1 += [cor1[0]]
        r2 += [cor2[0]]
        z1 += [z_transform(cor1[0], X_r.shape[0])]
        z2 += [z_transform(cor2[0], X_r.shape[0])]
        p1 += [cor1[1]]
        p2 += [cor2[1]]
    return np.median(r1), np.median(z1), np.median(p1), \
        np.median(r2), np.median(z2), np.median(p2)


def dataset_correlation(X, Y):
    #r1, z1, p1, o1, r2, z2, p2, o2           = pls(X,Y)
    #cv_r1, cv_z1, cv_p1, cv_r2, cv_z2, cv_p2 = cross_validate(X, Y)
    #res = pls(X,Y)
    cv = cross_validate(X, Y)
    return cv
