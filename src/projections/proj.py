# Imports

import h5py
import json
import uuid

import sys, os
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import numpy as np
import string
import random
import subprocess
import checkerconfig


from auto_plots import vector_validation, projection_plot
from checkerUtils import all_coords,coordinate2mosaic

import collections
from scipy import spatial

import argparse



# Variables

parser = argparse.ArgumentParser()

parser.add_argument('--manifold', default = 'tsne', type=str, help = 'largevis, tsne, mds')
parser.add_argument('--filename', default = 'clustemb.h5', type=str, help = 'Signature file.')
parser.add_argument('--table', default = None, type=str, help = 'MOSAIC table name.')
parser.add_argument('--max_comp', default = None, type=int, help = 'Maximum number of components to account for in the original space.')
parser.add_argument('--dev', default=0, type=float, help = 'STD deviation in the 2D projection, to prettify.')
parser.add_argument('--threads', default = -1, type = int, help = 'Number of training threads')
parser.add_argument('--samples', default = -1, type = int, help = 'Number of training mini-batches')
parser.add_argument('--prop', default = -1, type = int, help = 'Number of propagations')
parser.add_argument('--alpha', default = -1, type = float, help = 'Learning rate')
parser.add_argument('--trees', default = -1, type = int, help = 'Number of rp-trees')
parser.add_argument('--neg', default = -1, type = int, help = 'Number of negative samples')
parser.add_argument('--neigh', default = None, type = int, help = 'Number of neighbors in the NN-graph')
parser.add_argument('--gamma', default = -1, type = float, help = 'Weight assigned to negative edges')
parser.add_argument('--perp', default = -1, type = int, help = 'Perplexity for the NN-grapn')
parser.add_argument('--theta', default = 0.5, type = float, help = 'Theta in BH-TSNE, from 0 to 0.5')
parser.add_argument('--transparency', default = 0.3, type = float, help = 'Transparency in the projection plot')
parser.add_argument('--s', default = None, type = float, help = 'Size of the dot')
parser.add_argument('--bw', default = None, type = float, help = "Bandwidth of the KDE")
parser.add_argument('--levels', default = 10, type = int, help = "Levels of the contour plot")
parser.add_argument('--local_run', default = False,  help = "Define if this a local run of the script or path of the pipeline")
parser.add_argument('--only_plot', default = False, action = 'store_true', help = "Only do the plots, it needs proj.tsv to be pre-calculated")
parser.add_argument('--unique', default = False, action = 'store_true', help = "Project only on unique samples")
parser.add_argument('--plots_folder', default = None, type = str, help = 'Plots folder')
parser.add_argument('--filesdir', default = None, type = str, help = "Where validation files are stored")


args = parser.parse_args()

print args


if args.table in checkerconfig.TABLE_COORDINATES:
    coord = checkerconfig.TABLE_COORDINATES[args.table]
else:
    sys.exit("Table %s is not know to the Chemical Checker...!" % args.table)

if args.plots_folder is None:
    args.plots_folder = coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_PLOTS_FOLDER
    
if args.local_run == False:
    savedir = coordinate2mosaic(coord) + "/" 
else:
    savedir = ""
# Start

if args.only_plot:
    with h5py.File(savedir + "proj.h5", "r") as hf:
        Proj = hf["V"][:]
    projection_plot(args.table, Proj, bw = args.bw, levels = args.levels, plot_folder = args.plots_folder)
    sys.exit("Plots done")

V = None

if args.unique:
    drows = collections.defaultdict(list)    
    with h5py.File(savedir + args.filename, "r") as hf:
        inchikeys = hf["keys"][:]
        V = hf["V"][:]
        for i in xrange(len(inchikeys)):
            drows[tuple(V[i])] += [i]
        V = np.vstack(x for x in drows.keys())
    print "Repeats removed. Before = %d After = %d" % (len(inchikeys), V.shape[0]) 
    traceback = np.zeros(len(inchikeys), dtype = int)
    for i in xrange(V.shape[0]):
        for j in drows[tuple(V[i])]:
            traceback[j] = i

if args.manifold == "largevis":

    if not args.unique:
        hf = h5py.File(savedir + args.filename, "r")
        inchikeys = hf["keys"][:]
        V = hf["V"]

    from largevis import largevis

    tmp = str(uuid.uuid4())
    largevis_file = tmp + ".txt"

    def largevis_file_from_file(filename, max_comp = None):
        
        faux = open(savedir + largevis_file + "aux", "w")
        c = 0
   
        if args.unique:
            N = V.shape[0]
        else:
            N = len(inchikeys)

        for i in xrange(N):
            v = V[i]
            if max_comp:
                v = v[:max_comp]
            s = " ".join(np.char.mod('%.5f', v))
            faux.write(s + "\n")
            c += 1
        faux.close()
        
        f = open(savedir + largevis_file, "w")
        f.write("%d %d\n" % (c, len(s.split(" "))))
        faux = open(savedir + largevis_file + "aux", "r")
        for l in faux:
            f.write(l)
        faux.close()
        f.close()
        os.remove(savedir + largevis_file + "aux")
        return c

    N = largevis_file_from_file(args.filename, max_comp = args.max_comp)

    if not args.neigh:
        neigh = np.max([30, np.min([150, int(np.sqrt(N)/2)])])
    else:
        neigh = args.neigh

    if not args.unique:
        hf.close()

    Proj = largevis(largevis_file, threads = args.threads, samples = args.samples, prop = args.prop, neigh = neigh, perp = args.perp, alpha = args.alpha, gamma = args.gamma, trees = args.trees, neg = args.neg)

    if args.unique:
        AuxProj = []
        for i in traceback:
            AuxProj += [Proj[i]]
        Proj = np.array(AuxProj)

    # Save to a file

    with h5py.File(savedir + "proj.h5", "w") as hf:

        if args.unique:
            pass    

        inchikey_proj = {}
        i = 0
        for inchikey in inchikeys:
            inchikey_proj[inchikey] = Proj[i,]
            i += 1
        inchikeys = np.array(inchikeys)
        hf.create_dataset("keys", data = inchikeys)
        hf.create_dataset("V", data = Proj)

    # Clean

    subprocess.Popen("rm %s annoy_index_file" % (savedir + largevis_file), shell = True).wait()

else:

    if not args.unique:
        with h5py.File(savedir + args.filename, "r") as hf:
            inchikeys = hf["keys"][:]
            V = hf["V"][:]

    if args.manifold == "tsne":

        from MulticoreTSNE import MulticoreTSNE as TSNE

        if args.perp == -1:
            neigh = np.max([30, np.min([150, int(np.sqrt(V.shape[0])/2)])])
            perp = int(neigh / 3)
        else:
            perp = args.perp
        tsne = TSNE(n_jobs = 4, perplexity = perp, angle = args.theta, n_iter = 1000, metric = "cosine")
        Proj = tsne.fit_transform(V.astype(np.float64))
        V = []
    
    elif args.manifold == "mds":

        from sklearn.manifold import MDS

        mds = MDS(n_jobs = -1)
        Proj = mds.fit_transform(V.astype(np.float64))

    else:
        sys.exit("Unrecognized manifold %s" % args.manifold)

    if args.unique:
        AuxProj = []
        for i in traceback:
            AuxProj += [Proj[i]]
        Proj = np.array(AuxProj)

    # Save to a file
    
    with h5py.File(savedir + "proj.h5", "w") as hf:
        inchikey_proj = {}
        for i in xrange(len(inchikeys)):
            k = inchikeys[i]
            inchikey_proj[k] = Proj[i]
        hf.create_dataset("V", data = Proj)
        hf.create_dataset("keys", data = inchikeys)

# Plot

xlim, ylim = projection_plot(args.table, Proj, bw = args.bw, levels = args.levels,plot_folder = args.plots_folder)


# MOA validation

print "Doing MoA validation"

ks_moa, auc_moa = vector_validation(inchikey_proj, "proj", args.table, prefix = "moa",plot_folder = args.plots_folder,files_folder = args.filesdir)
ks_atc, auc_atc = vector_validation(inchikey_proj, "proj", args.table, prefix = "atc",plot_folder = args.plots_folder,files_folder = args.filesdir)


# Saving results

INFO = {
"molecules": Proj.shape[0],
"moa_ks_d": ks_moa[0],
"moa_ks_p": ks_moa[1],
"moa_auc": auc_moa,
"atc_ks_d": ks_atc[0],
"atc_ks_p": ks_atc[1],
"atc_auc": auc_atc,
"xlim": xlim,
"ylim": ylim
}

with open(savedir + 'proj_stats.json', 'w') as fp:
    json.dump(INFO, fp)
