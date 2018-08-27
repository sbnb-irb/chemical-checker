#!/miniconda/bin/python


import gc
import h5py
import json
import math
import uuid
from tqdm import tqdm
from gensim import corpora, models
import collections
import sys, os, glob
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0],"../../pipeline/config"))
import Psql
import checkerconfig
from checkerUtils import logSystem,log_data,tqdm_local,coordinate2mosaic
from auto_plots import variance_plot, vector_validation, matrix_plot, euclidean_background
import numpy as np
import collections
import subprocess
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils.sparsefuncs import mean_variance_axis
import random
import shelve
import argparse
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import csv

# Variables

random.seed(42)



# Corpus class

class MyCorpus(object):
    
    def __init__(self, plain_corpus,dictionary):
        self.plain_corpus = plain_corpus
        self.dictionary = dictionary
    
    def __iter__(self):
        for l in open(self.plain_corpus, "r"):
            l = l.rstrip("\n").split(" ")[1].split(",")
            bow = self.dictionary.doc2bow(l)
            if not bow:
                continue
            yield bow
    
    def __len__(self):
        return len([_ for _ in self.inchikeys()])
    
    def inchikeys(self):
        for l in open(self.plain_corpus, "r"):
            inchikey = l.split(" ")[0]
            l = l.rstrip("\n").split(" ")[1].split(",")
            bow = self.dictionary.doc2bow(l)
            if not bow:
                continue
            yield inchikey

# Small functions

def integerize(V,recycle,models_folder):

    def callibrator(lb, ub):
        # Returns a*x + b to convert V to an integer scale
        a = float(255) / (ub - lb)
        b = 127 - a*ub
        return a, b
    
    # Convert to integer type from -128 to 127
    if not recycle or not os.path.exists(models_folder+"/integerizer_ab.txt"):
        lb = np.min(V)
        ub = np.max(V)
        a, b = callibrator(lb, ub)
        with open(models_folder+"/integerizer_ab.txt", "w") as f:
            f.write("%f\n%f" % (a,b))
    else:
        with open(models_folder+"/integerizer_ab.txt", "r") as f:
            a, b = [float(x) for x in f.read().split("\n")]
    def callibration(x): return a*x + b
    V = callibration(V)
    V = V.astype(np.int8)
    
    return V

        

def lsi_variance_explained(tfidf_corpus, lsi, B , N , num_topics,log=None):

        # Variance estimation (this may take a while...)
        # B: Number of runs, to ensure robustness
        # N: Size of the random sample sample (1000 should be enough, 100 works)

        mm = corpora.MmCorpus(tfidf_corpus)

        exp_var_ratios = []
        for _ in tqdm_local(log,xrange(B)):
            xt = []
            sm = lil_matrix((N, mm.num_terms))
            #sm = csr_matrix((N, mm.num_terms))
            for i in xrange(N):
                io = random.randint(0, mm.num_docs - 1)
                terms = mm[io]
                # Transformed matrix
                tops = np.zeros(num_topics)
                for x in lsi[terms]:
                    if x[0] >= num_topics: continue
                    tops[x[0]] = x[1]
                xt += [tops]
                # Sparse original matrix
                for t in terms:
                    sm[i, t[0]-1] = t[1]
            xt = np.array(xt)
            sm = sm.tocsr()
            full_var = mean_variance_axis(sm, 0)[1].sum()

            try:
                exp_var = np.var(xt, axis=0)
                exp_var_ratios += [exp_var / full_var]
            except:
                continue

        exp_var_ratios = np.mean(np.array(exp_var_ratios), axis = 0)

        return exp_var_ratios

def generate_signatures(dbname, table,infile = None,outfile = None,models_folder = None,plots_folder = None,sig_stats_file = None, 
                       min_freq = 5, max_freq = None,num_topics = None, B = 10, N = 1000, B_euclidean =1000000, multipass = False,
                        recycle = False,variance_cutoff = 0.9, log = None, tmpDir = ''):
    

    checkercfg = checkerconfig.checkerConf()
    
    if table in checkerconfig.TABLE_COORDINATES:
        coord = checkerconfig.TABLE_COORDINATES[table]
    else:
        log_data(log,"Table %s is not know to the Chemical Checker...!" % table)
        return
        
    all_tables = checkercfg.getTableList("all")
    
    if table not in all_tables:
        log_data(log,"Unrecognized table %s" % table) 
        return
    
    log_data(log, "WORKING ON %s (%s)" % (coord, table))

    if outfile is None:
        outfile = coordinate2mosaic(coord) + "/" + checkerconfig.SIG_FILENAME
    
    if models_folder is None:
        models_folder = coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_MODELS_FOLDER
    
    if plots_folder is None:
        plots_folder = coordinate2mosaic(coord) + "/" + checkerconfig.DEFAULT_PLOTS_FOLDER
    
    if sig_stats_file is None:
        sig_stats_file = outfile.split(".h5")[0]+"_stats.json"
    
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)
    if not os.path.exists(models_folder): os.makedirs(models_folder)
    
    # Some heuristics here...
    if max_freq is None:
        if table in ("fp2d", "fp3d", "scaffolds"):
            max_freq = 0.8
        elif table == "subskeys":
            max_freq = 0.9
        else:
            max_freq = 0.25
    
    if tmpDir != '':
        tmp =  os.path.join(tmpDir,str(uuid.uuid4()))
    else:
        tmp = str(uuid.uuid4())

    # In the chemistry level, restrict search.
    
    mymols = set()
    
    
    if table in checkerconfig.CHEM_TABLES :
        
    
        all_chemistry = checkercfg.getTableList("chemistry")
        tables = list(set(all_tables) - set(all_chemistry))
        
        for t in tqdm_local(log,tables): mymols.update(Psql.fetch_inchikeys(t,dbname))    
    
    # LSI
    
    if table not in checkerconfig.CONTINUOUS_TABLES:
        # Temporary files
    
        plain_corpus = tmp+".corpus.txt"
    
        tfidf_corpus = tmp+".mm"
    
        # Write corpus (inchikey sig)
    
        InitMols = 0
    
        if table in checkerconfig.CHEM_TABLES:
    
            # Fetch and retain only those molecules
    
            f = open(plain_corpus, "w")
    
            if not infile:
    
                query = "SELECT inchikey, raw FROM %s WHERE raw IS NOT NULL" % table
    
                con = Psql.connect(dbname)
                cur = con.cursor()
                cur.execute(query)
                for r in tqdm_local(log,cur):
                    if r[0] not in mymols: continue
                    f.write("%s %s\n" % (r[0], r[1]))
                    InitMols += 1
    
            else:
                
                with open(infile, "r") as ff:
                    
                    for r in csv.reader(ff, delimiter = "\t"):
                        f.write("%s %s\n" % (r[0], r[1]))
                        InitMols += 1
            
            f.close()
    
    
        elif table in checkerconfig.CATEG_WEIGHTED_TABLES:          
            # These data have (1), (2)...
    
            f = open(plain_corpus, "w")
    
            if not infile:
    
                for r in tqdm_local(log,Psql.qstring("SELECT inchikey, raw FROM %s WHERE raw IS NOT NULL" % table, dbname)):
                    s = r[1].split(",")
                    sd = []
                    for x in s:
                        d = int(x.split("(")[1].split(")")[0])
                        sd += d*[x.split("(")[0]]
                    sd = ",".join(sd)
                    f.write("%s %s\n" % (r[0], sd))
                    InitMols += 1
    
            else:
    
                with open(infile, "r") as ff:
    
                    for r in csv.reader(ff, delimiter = "\t"):
                        s = r[1].split(",")
                        sd = []
                        for x in s:
                            d = int(x.split("(")[1].split(")")[0])
                            sd += d*[x.split("(")[0]]
                        sd = ",".join(sd)
                        f.write("%s %s\n" % (r[0], sd))
                        InitMols += 1
    
            f.close()
    
        else:
            f = open(plain_corpus, "w")
    
            if not infile:
            
                for r in tqdm_local(log,Psql.qstring("SELECT inchikey, raw FROM %s WHERE raw IS NOT NULL" % table, dbname)):
                    f.write("%s %s\n" % (r[0], r[1]))
                    InitMols += 1
    
            else:
    
                with open(infile, "r") as ff:
    
                    for r in csv.reader(ff, delimiter = "\t"):
                        f.write("%s %s\n" % (r[0], r[1]))
                        InitMols += 1
    
            f.close()
    
        # Dictionary
    
        log_data(log, "Getting dictionary")
    
        if not recycle or not os.path.exists(models_folder+"/dictionary.pkl"):
    
            dictionary = corpora.Dictionary(l.rstrip("\n").split(" ")[1].split(",") for l in tqdm_local(log,open(plain_corpus, "r")))
    
            dictionary.filter_extremes(no_below=min_freq, no_above=max_freq)
            dictionary.compactify()
    
            dictionary.save(models_folder+"/dictionary.pkl")
    
        else:
        
            dictionary = corpora.Dictionary.load(models_folder+"/dictionary.pkl")
    
    
        log_data(log, "Terms: %d" % len(dictionary))
    
        # Functions to perform LSI
    
        c = MyCorpus(plain_corpus,dictionary)
    
        Mols = len(c)
        
        log_data(log, "Corpus length: " + str(Mols))
       
        if not recycle or not os.path.exists(models_folder+"/tfidf.pkl"):
    
            log_data(log, "Calculating TFIDF model")
     
            tfidf = models.TfidfModel(c)
    
            tfidf.save(models_folder+"/tfidf.pkl")
    
        else:
    
            log_data(log, "Loading TFIDF model")
    
            tfidf = models.TfidfModel.load(models_folder+"/tfidf.pkl")
    
        c_tfidf = tfidf[c]
        
        log_data(log, "Saving TFIDF corpus, indexed")
        
        corpora.MmCorpus.serialize(tfidf_corpus, c_tfidf)
    
        if not recycle or not os.path.exists(models_folder+"/lsi.pkl"):
            
            log_data(log, "LSI model with %d topics..." % num_topics)
    
            if multipass:
                onepass = False
            else:
                onepass = True
    
            lsi = models.LsiModel(c_tfidf, id2word = dictionary, num_topics = num_topics, onepass = onepass)
    
            lsi.save(models_folder+"/lsi.pkl")
    
        else:
    
            lsi = models.LsiModel.load(models_folder+"/lsi.pkl")
    
        log_data(log, "LSI transformation of the TF-IDF corpus...")
    
        c_lsi = lsi[c_tfidf]    
    
        # Decide number of topics
    
        if not recycle or not os.path.exists(models_folder+"/cut.txt"):
         
            log_data(log, "Deciding number of topics")
    
            exp_var_ratios = lsi_variance_explained(tfidf_corpus, lsi, B = B, N = N, num_topics = num_topics,log = log)
    
            cut_i, elb_i = variance_plot(exp_var_ratios, table = table, variance_cutoff = variance_cutoff, plot_folder = plots_folder)
    
            with open(models_folder+"/cut.txt", "w") as f:
                f.write("%d\n%d\n" % (cut_i, elb_i))
    
        else:
    
            log_data(log, "Reading number of topics")
        
            with open(models_folder+"/cut.txt", "r") as f:
                cut_i = int(f.next().rstrip())
                elb_i = int(f.next().rstrip())
    
        log_data(log, "%.1f topics: %d" % (variance_cutoff, cut_i+1))
        log_data(log, "Elbow topics: %d" % (elb_i+1))
    
        # Get inchikeys
    
        inchikeys = np.array([k for k in c.inchikeys()])
    
        log_data(log, "Saving to %s" % outfile)
    
        V = np.empty((len(inchikeys), cut_i+1))
    
        i = 0
        for l in c_lsi:
            v = np.zeros(cut_i+1)
            for x in l[:cut_i+1]:
                if x[0] > cut_i: continue
                v[x[0]] = x[1]
            k = inchikeys[i]
            V[i,:] = v
            i += 1
    
        V = integerize(V,recycle,models_folder)
     
        inchikey_sig = shelve.open(tmp+".dict", "n")
        for i in tqdm_local(log,xrange(len(inchikeys))):
            inchikey_sig[str(inchikeys[i])] = V[i]
        inchikey_sig.close()
        f.close()
    
        log_data(log, "... but sorting before!")
        sort_idxs = np.argsort(inchikeys)
    
        with h5py.File(outfile, 'w') as hf:
            hf.create_dataset("keys", data = inchikeys[sort_idxs])
            hf.create_dataset("V",  data = V[sort_idxs])
    
        V = None
        c_lsi = None
    
    # PCA
    
    else:
    
        if table == "physchem":
    
            if not infile:
    
                # Retain only molecules of interest.
                R = Psql.qstring("SELECT * FROM %s WHERE raw IS NOT NULL" % table, dbname)
                RowNames = []
                X = []
                for r in tqdm_local(log,R):
                    if r[0] not in mymols: continue
                    RowNames += [r[0]]
                    X += [r[1:]]
                X = np.array(X)
    
            else:
    
                with open(infile, "r") as f:
                    X = []
                    RowNames = []
                    for r in csv.reader(f, delimiter = "\t"):
                        RowNames += [r[0]]
                        X += [r[1:]]
                    X = np.array(X)
    
            if not recycle or not os.path.exists(models_folder+"/robustscaler.pkl"):
                # Scale
                scl = RobustScaler()
                scl.fit(X)
                joblib.dump(scl, models_folder+"/robustscaler.pkl")
            else:
                scl = joblib.load(models_folder+"/robustscaler.pkl")
    
            X = scl.transform(X)
    
            scl = None
    
        else:
    
            if not infile:
                R = Psql.qstring("SELECT inchikey, raw FROM %s WHERE raw IS NOT NULL" % table, dbname)
                RowNames = []
                X = []
                for r in tqdm_local(log,R):
                    RowNames += [r[0]]
                    X += [[float(x) for x in r[1].split(",")]]
                X = np.array(X)
            else:
                with open(infile, "r") as f:
                    RowNames = []
                    X = []
                    for r in csv.reader(f, delimiter = "\t"):
                        RowNames += [r[0]]
                        X += [[float(x) for x in r[1].split(",")]]
                    X = np.array(X)
    
        InitMols = X.shape[0]
        Mols = InitMols
    
        if not recycle or not os.path.exists(models_folder+"/pca.pkl"):
    
            log_data(log, "Fitting PCA")
    
            pca = PCA(n_components = variance_cutoff)
    
            pca.fit(X)
    
            joblib.dump(pca, models_folder+"/pca.pkl")
    
        else:
    
            pca = joblib.load(models_folder+"/pca.pkl")
    
        if not recycle or not os.path.exists(models_folder+"/cut.txt"):
    
            log_data(log, "Looking for variance")
            cut_i, elb_i = variance_plot(pca.explained_variance_ratio_, table = table, variance_cutoff = variance_cutoff, plot_folder = plots_folder)
            with open(models_folder+"/cut.txt", "w") as f:
                f.write("%d\n%d\n" % (cut_i, elb_i))
    
        else:
            # Not really necessary, but done to be coherent with LSI.
            log_data(log, "Reading number of topics")
            with open(models_folder+"/cut.txt", "r") as f:
                cut_i = int(f.next().rstrip())
                elb_i = int(f.next().rstrip())
    
        log_data(log, "Projecting")
    
        V = pca.transform(X)
    
        log_data(log, "Saving stuff")
    
        V = integerize(V,recycle,models_folder)
    
        inchikeys = []
        inchikey_sig = shelve.open(tmp+".dict", "n")
        for i in tqdm_local(log,xrange(len(RowNames))):
            inchikey = RowNames[i]
            inchikeys += [inchikey]
            inchikey_sig[str(inchikey)] = V[i]
        inchikey_sig.close()
        inchikeys = np.array(inchikeys)
    
        log_data(log, "... but sorting before!")
        sort_idxs = np.argsort(inchikeys)
    
        with h5py.File(outfile, "w") as hf:
            hf.create_dataset("keys", data = inchikeys[sort_idxs])
            hf.create_dataset("V", data = V[sort_idxs])
        
        V = []    
        pca = []
    
 
    
    
    # Euclidean_distance significances
    
    if not recycle or not os.path.exists(models_folder+"/bg_euclideans.h5"):
    
        log_data(log, "Computing euclidean distance empirical P-values")
    
        inchikey_sig = shelve.open(tmp+".dict", "r")
        pvals = euclidean_background(inchikey_sig, inchikeys, B = B_euclidean)
        inchikey_sig.close()
    
        with h5py.File(models_folder+"/bg_euclideans.h5", "w") as hf:
            hf.create_dataset("distance", data = np.array([p[0] for p in pvals]))
            hf.create_dataset("pvalue"  , data = np.array([p[1] for p in pvals]))
            hf.create_dataset("integer" , data = np.array([p[2] for p in pvals]).astype(np.int8))
    
    log_data(log, "Cleaning")
    gc.collect()
    try:
        os.remove(plain_corpus)
        os.remove(tfidf_corpus)
        os.remove(tfidf_corpus+".index")
    except:
        pass
    
    # Validation
    
    if infile is not None:
        os.remove(tmp+".dict")
        log_data(log,"Since a file was inputted, no MoA/ATC validation is done!")
        return
    
    log_data(log, "MOA and ATC Validations")
    
    inchikey_sig = shelve.open(tmp+".dict", "r")
    ks_moa, auc_moa = vector_validation(inchikey_sig, "sig", table, prefix = "moa", plot_folder = plots_folder, files_folder = tmpDir )
    ks_atc, auc_atc = vector_validation(inchikey_sig, "sig", table, prefix = "atc", plot_folder = plots_folder, files_folder = tmpDir )
    inchikey_sig.close()
    for filename in glob.glob(tmp+".dict*") :
        os.remove(filename)
    
    # Cleaning
    
    log_data(log, "Matrix plot")
    
    matrix_plot(table,coordinate2mosaic(coord),plots_folder)
    
    # Statistics file
    
    log_data(log,  "Statistics file")
    
    INFO = {
    "initial_molecules": InitMols,
    "molecules": Mols,
    "latent_variables": (cut_i + 1),
    "elbow_variables": (elb_i + 1),
    "moa_ks_d": ks_moa[0],
    "moa_ks_p": ks_moa[1],
    "moa_auroc": auc_moa,
    "atc_ks_d": ks_atc[0],
    "atc_ks_p": ks_atc[1],
    "atc_auroc": auc_atc,
    "min_freq": min_freq,
    "max_freq": max_freq,
    "num_topics": num_topics,
    "variance_cutoff": variance_cutoff
    }
    
    with open(coordinate2mosaic(coord) + "/"+'sig_stats.json', 'w') as fp:
        json.dump(INFO, fp)
        
if __name__ == '__main__':


    # Parse arguments

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--db', type = str, help = 'Chemical Checker database name', required = True)
    parser.add_argument('--table', type = str, help = 'Chemical Checker table name', required = True)
    parser.add_argument('--infile', default = None, type = str, help = 'Raw signature file (should coincide with formats available in the database)')
    parser.add_argument('--outfile', default = None, type = str, help = 'Output file with extension .h5')
    parser.add_argument('--models_folder', default = None, type = str, help = 'Models folder')
    parser.add_argument('--plots_folder', default = None, type = str, help = 'Plots folder')
    parser.add_argument('--sig_stats_file', default = None, type = str, help = 'Signature stats file')
    parser.add_argument('--min_freq', default = 5, type = int, help = 'Minimum frequency (counts)')
    parser.add_argument('--max_freq', default = None, type = float, help = 'Maximim frequency (proportion')
    parser.add_argument('--num_topics', default = None, type = int, help = 'Number of topics to start with')
    parser.add_argument('--B', default = 10, type = int, help = 'In the variance explained, number of random passes')
    parser.add_argument('--N', default = 1000, type = int, help = 'In the variance explained, number of random samples')
    parser.add_argument('--B_euclidean', default = 1000000, type = int, help = 'In the euclidean distance estimation, number of random pairs')
    parser.add_argument('--multipass', default = False, action = 'store_true', help = 'Multi-pass, for large datasets')
    parser.add_argument('--recycle', default = False, action = 'store_true', help = 'Recycle stored models')
    parser.add_argument('--variance_cutoff', default = 0.9, type = float, help = "Variance cutoff")
    
    args = parser.parse_args()

    
    generate_signatures(args.db, args.table,args.infile,args.outfile,args.models_folder,arg.plots_folder,args.sig_stats_file,args.min_freq,
                        args.max_freq,args.num_topics,args.B,args.B_euclidean,args.multipass,args.recycle,args.variance_cutoff,None)

