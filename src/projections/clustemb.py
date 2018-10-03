'''

Siamese network to predict or classify clusters.

It should be enough to organize in 2D, later on.

'''

# Imports

from __future__ import absolute_import
import argparse
import uuid
import numpy as np
np.random.seed(42)
import pandas as pd
import random
random.seed(42)
import os
import glob
import sys

from keras import backend as K
def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        os.environ["OMP_NUM_THREADS"]='1'
        os.environ["MKL_THREADING_LAYER"]="GNU"
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop, Adam

from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.losses
import theano.ifelse

import sys
import json
import bisect
from collections import Counter, defaultdict
import h5py

import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


import shelve
import collections

sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
from checkerUtils import all_coords,coordinate2mosaic
from auto_plots import vector_validation
import checkerconfig



# Variables

parser = argparse.ArgumentParser()

parser.add_argument('--table', default = None, type = str, help = "Table")
parser.add_argument('--infile', default = None, type = str, help = "Matrix to embed")
parser.add_argument('--clust_infile', default = None, type = str, help = "Clusters")
parser.add_argument('--outfile', default = None, type = str, help = 'Output file with extension .h5')
parser.add_argument('--layers', default = None, type = str, help = "Layers, comma-separated")
parser.add_argument('--models_folder', default = None, type = str, help = 'Models folder')
parser.add_argument('--plots_folder', default = None, type = str, help = 'Plots folder')
parser.add_argument('--dropout', default = 0.2, type = float, help = "Dropout")
parser.add_argument('--epochs', default = 50, type = int, help = "Epochs to train")
parser.add_argument('--batch_size', default = 128, type = int, help = "Batch size")
parser.add_argument('--pairs_per_mol', default = 10000, type = int, help = "Number of positive pairs per molecule")
parser.add_argument('--max_pairs_per_cluster', default = 100000, type = int, help = "Maximum number of pairs per cluster")
parser.add_argument('--not_knn', default = False, action = 'store_true', help = "Do not do knn")
parser.add_argument('--knn', default = None, help = "In the k-nearest neighbors search")
parser.add_argument('--pvalue_cutoff', default = 0.05, help = "P-value cutoff in the background Euclideans")
parser.add_argument('--goal_pairs', default = 1000000, help = "Number of positive pairs that we have as a goal")
parser.add_argument('--recycle', default = False, action = 'store_true', help = "Recycle stored models")
parser.add_argument('--filesdir', default = None, type = str, help = "Where validation files are stored")
parser.add_argument('--max_dim', default = None, type = int, help = "Max dimentions to take from signatures")

args = parser.parse_args()

tmp = str(uuid.uuid4())

print args

# Preparation

if args.table in checkerconfig.TABLE_COORDINATES:
    coord = checkerconfig.TABLE_COORDINATES[args.table]
else:
    sys.exit("Table %s is not know to the Chemical Checker...!" % args.table)


if args.outfile is None:
    args.outfile = coordinate2mosaic(coord) + "/" + checkerconfig.CLUSTSIAM_FILENAME

if args.models_folder is None:
    args.models_folder = coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_MODELS_FOLDER

if args.plots_folder is None:
    args.plots_folder = coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_PLOTS_FOLDER

if args.infile is None and args.clust_infile is None:
    infile = coordinate2mosaic(coord) + "/" + checkerconfig.SIG_FILENAME
    clust_infile = coordinate2mosaic(coord) + "/" + checkerconfig.CLUST_FILENAME
else:
    if args.infile is None: sys.exit("Please make sure that clusters and signatures coincide")
    if args.clust_infile is None: sys.exti("Please make sure thatn clusters and signatures coincide")
    infile = args.infile
    clust_infile = args.clust_infile

if not os.path.exists(args.plots_folder): os.mkdir(args.plots_folder)
if not os.path.exists(args.models_folder): os.mkdir(args.models_folder)

model_file = args.models_folder + "/clustemb_network.h5"


print "WORKING ON %s (%s)" % (coord, args.table)

print "Reading data"

with h5py.File(infile, "r") as hf:
    inchikeys = hf["keys"][:]
    if not args.max_dim:
        V = hf["V"][:]
    else:
        V = hf["V"][:,:args.max_dim]

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                    (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

keras.losses.custom_loss = contrastive_loss

if not args.recycle:

    # Euclidean distance cutoff
    with h5py.File(args.models_folder + "/bg_euclideans.h5") as hf:
        pvals = hf["pvalue"][:]
        dists = hf["distance"][:]
        dist_cut = dists[bisect.bisect_left(pvals, args.pvalue_cutoff)]

    with h5py.File(clust_infile, "r") as hf:
        labels = hf["labels"][:]

    print "Creating pairs of positive and negative samples"

    def produce_pairs(labels):

        clusts = sorted(set(labels))

        # Infer the number of pairs, in case it is necessary
        if not args.not_knn:
            if args.knn is None:
                pairs_per_cluster = int(args.goal_pairs / len(clusts))
                ideal_k = int(np.sqrt(pairs_per_cluster * 2))
                if ideal_k > 30:
                    n_neighbors = 30
                elif ideal_k < 5:
                    n_neighbors = 5
                else:
                    n_neihbors = ideal_k
            else:
                n_neighbors = args.knn

            print "Taking %d neighbors inside each cluster" % n_neighbors

        pairs  = []
        Y      = []
        for clust in clusts:
            
            # Indices in the cluster and outside of it
            pos = np.where(labels == clust)[0]
            neg = np.where(labels != clust)[0]

            # Positive pairs
            if args.not_knn:
                N = np.min([int(len(pos)*(len(pos)-1)/2.), args.pairs_per_mol * len(pos), args.max_pairs_per_cluster])
                pos_pairs = set()
                while len(pos_pairs) < N:
                    p = random.sample(pos, 2)
                    pos_pairs.update([tuple(sorted(p))])
            else:
                knn = NearestNeighbors(n_neighbors = np.min([n_neighbors, len(pos)]), metric = "euclidean")
                knn.fit(V[pos])
                dists, neighs = knn.kneighbors(V[pos])
                pos_pairs = set()
                for i, ns in enumerate(neighs):
                    for j in ns[dists[i] <= dist_cut]:
                        pos_pairs.update([tuple(sorted([pos[i], pos[j]]))])
                
            # Shuffle positive pairs
            posp = []
            for p in pos_pairs:
                p = list(p)
                random.shuffle(p)
                posp += [p]
            
            # Now the negative pairs
            neg_pairs = set()
            while len(neg_pairs) < len(posp):
                p = random.choice(posp)[random.randint(0,1)]
                n = random.choice(neg)
                neg_pairs.update([(p, n)])

            negp = []
            neg_pairs = list(neg_pairs)
            random.shuffle(neg_pairs)
            for p in neg_pairs:
                p = list(p)
                random.shuffle(p)
                negp += [p]

            for i in xrange(len(posp)):
                pairs += [posp[i], negp[i]]
                Y += [1, 0]

        X, Y = np.array(pairs), np.array(Y)

        idxs = [i for i in xrange(X.shape[0]/2)]
        np.random.shuffle(idxs)
        matrixX = []
        matrixY = []
        for idx in idxs:
            matrixX += [X[idx], X[idx+1]]
            matrixY += [Y[idx], Y[idx+1]]
        matrixX = np.array(matrixX)
        matrixY = np.array(matrixY)

        return matrixX, matrixY    

    matrixX, matrixY = produce_pairs(labels)

    print "Working on %d pairs" % matrixX.shape[0]

    # Network architecture

    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def architecture(input_dim):

        if not args.layers:
        
            if input_dim < 128:
                return [32]

            if input_dim < 256:
                return [64, 32]

            if input_dim < 512:
                return [128, 64, 32]

            return [256, 128, 64, 32]
        
        else:
        
            return [int(l) for l in args.layers.split(",")]


    def create_base_network(input_dim, arch_network):
        '''Base network to be shared (eq. to feature extraction).'''
        
        seq = Sequential()
        
        # Input layer
        seq.add(Dense(int(arch_network[0]), input_shape=(input_dim,), activation='relu'))
        seq.add(BatchNormalization())
        seq.add(Dropout(args.dropout))
        
        if len(arch_network) > 2:

            for l in arch_network[1:-1]:
                seq.add(Dense(l, activation = "relu"))
                seq.add(BatchNormalization())
                seq.add(Dropout(args.dropout))

        #seq.add(Dense(arch_network[-1], activation = "sigmoid"))
        seq.add(Dense(arch_network[-1]))

        return seq


    def compute_accuracy(predictions, labels):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return labels[predictions.ravel() < 0.5].mean()


    def batch_generator(matrixX, matrixY, batch_size = 100):
        #matrixX = X_train_pairs; matrixY = Y_train;
        while True:
            nsamples_train = matrixX.shape[0];
            nbatches = int(np.ceil(float(nsamples_train) / batch_size))

            for ib in range(nbatches):
                # Take the index of pairs of signatures for that batch size
                pairs_X = matrixX[ib*batch_size:(ib+1)*batch_size]
                pairs_Y = matrixY[ib*batch_size:(ib+1)*batch_size]

                X_left   = V[pairs_X[:,0],:]
                X_right  = V[pairs_X[:,1],:]
                Y_middle = np.array([xx for xx in pairs_Y])

                yield [X_left, X_right], Y_middle


    input_dim = V.shape[1]

    arch_network = architecture(input_dim)
    print "Architecture", arch_network
    base_network = create_base_network(input_dim,arch_network)
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    processed_a = base_network(input_a) 
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    adam = Adam();
    model.compile(loss=contrastive_loss, optimizer=adam)

    earlyStopping = EarlyStopping(monitor = 'loss', patience = 2, verbose = 0, mode = 'auto', min_delta = 0.001)

    steps_per_epoch = int(np.ceil(float(matrixX.shape[0]) / args.batch_size))

    print "Fitting"

    model.fit_generator(batch_generator(matrixX, matrixY, batch_size=args.batch_size), epochs=args.epochs, callbacks = [earlyStopping], steps_per_epoch=steps_per_epoch, verbose=1)
    model.save(model_file)

else:

    print "Loading model"
    model = load_model(model_file, custom_objects={'contrastive_loss': contrastive_loss})


def encoder(model, V):

    layer_name = 'sequential_1'
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).get_output_at(1))
    return intermediate_layer_model.predict([V,V])

print "Encoding"

E = encoder(model, V)

with h5py.File(args.outfile, "w") as hf:
    hf.create_dataset("V", data = E)
    hf.create_dataset("keys", data = inchikeys)

if args.infile is not None and args.clust_infile is not None:
    sys.exit("Since a file was inputted, no MoA/ATC validation is done!")

print "MOA and ATC Validations"

inchikey_emb = shelve.open(tmp+".dict", "n")
for i in xrange(len(inchikeys)):
    inchikey_emb[str(inchikeys[i])] = E[i]
ks_moa, auc_moa = vector_validation(inchikey_emb, "clustemb", args.table, prefix = "moa", plot_folder = args.plots_folder,files_folder = args.filesdir)
ks_atc, auc_atc = vector_validation(inchikey_emb, "clustemb", args.table, prefix = "atc", plot_folder = args.plots_folder,files_folder = args.filesdir)
inchikey_emb.close()

print "Cleaning"
for filename in glob.glob(tmp+".dict*") :
    os.remove(filename)
