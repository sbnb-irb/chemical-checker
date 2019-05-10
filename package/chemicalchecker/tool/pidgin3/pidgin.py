'''
A wrapper for PIDGIN v3 adapted to work within the Chemical Checker.

Author : Lewis Mervin lhm30@cam.ac.uk
Supervisor : Dr. A. Bender
All rights reserved 2018
Protein Target Prediction using on SAR data from PubChem and ChEMBL_24
Molecular Descriptors : 2048bit circular Binary Fingerprints (Rdkit) - ECFP_4

Output a matrix of probabilities [computed as the mean predicted class probabilities of
the trees in the forest (where the class probability of a single tree is the fraction of
samples of the same class in a leaf)], or user-specified Random probability thresholds to
produce binary predictions for an input list of smiles/sdfs. Predictions are generated
for the [filtered] models using a reliability-density neighbourhood Applicability Domain
(AD) analysis from: doi.org/10.1186/s13321-016-0182-y
'''

#standard libraries
import bz2
import cPickle
import csv
import glob
import math
import multiprocessing
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from optparse import OptionParser
from os import path
import collections

#third-party libraries
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from scipy.stats import percentileofscore
from standardiser import standardise

# CC libraries
from chemicalchecker.util import logged
from chemicalchecker.util import Config
from chemicalchecker.util.parser import Converter

# Variables

sep = "/"
threshold = None

# Functions

# Get models
@logged
def get_Models(pdg):
    target_count = 0
    model_info = [l.split('\t') for l in open(pdg.mod_dir + '/training_log.txt').read().splitlines()]
    if pdg.ntrees: model_info = [m[:2]+[str(pdg.ntrees)]+m[3:] if idx > 0 else m \
    for idx, m in enumerate(model_info)]
    model_info = {l[0] : l for l in model_info}
    uniprot_info = [i.split('\t') for i in open(pdg.mod_dir + '/uniprot_information.txt').read().splitlines()[1:]]
    mid_uniprots = dict()
    if pdg.p_filt:
        val_dict = {'tsscv':7, 'l50po':12, 'l50so':17}
        metric_dict = dict(zip(['bedroc','roc','prauc','brier'],range(4)))
        try:
            validation, metric, perf_threshold = [str(x) for x in pdg.p_filt]
            train_row_idx = val_dict[validation] + metric_dict[metric]
            perf_threshold = float(perf_threshold)
            get_Models._log.debug(' Filtering models for a minimum ' + metric + ' performance of ' \
             + str(perf_threshold) + ' during ' + validation + ' validaiton')
        except (KeyError,ValueError):
            get_Models._log.debug(' Input Error [--performance_filter]: Use format ' \
            'validation_set[tsscv,l50so,l50po],metric[bedroc,roc,prauc,brier],' \
            'threshold[float].\n E.g "bedroc,tsscv,0.5"\n...exiting')
            sys.exit()
    if pdg.organism: orgs = map(str.lower, pdg.organism)
    for row in uniprot_info:
        #filter bioactivity/organism/targetclass/minsize/se/performance (if set)
        if pdg.bioactivity and float(row[8]) not in pdg.bioactivity: continue
        if pdg.organism and (row[4] == '' \
        or not any([org.lstrip() in row[4].lower() for org in orgs])): continue
        if pdg.targetclass and row[3] not in pdg.targetclass: continue
        if sum(map(int,row[9:11])) < pdg.minsize: continue
        if pdg.se_filt and int(row[13]) > 0: continue
        if pdg.p_filt:
            try:
                if float(model_info[row[-1]][train_row_idx].split(',')[0]) < perf_threshold: continue
            except ValueError: continue
        #if passes all filters then add to mid->uniprot dictionary
        try: mid_uniprots[row[-1]].append(row)
        except KeyError: mid_uniprots[row[-1]] = [row]
        target_count +=1
    if len(mid_uniprots) == 0:
        get_Models._log.warning('No eligible models using current filters...exiting')
        sys.exit()
    return mid_uniprots, model_info, target_count

#preprocess rdkit molecule
def preprocessMolecule(inp):
    def checkC(mm):
        mwt = Descriptors.MolWt(mm)
        for atom in mm.GetAtoms():
            if atom.GetAtomicNum() == 6 and 100 <= mwt <= 1000: return True
        return False
    def checkHm(mm):
        for atom in mm.GetAtoms():
            if atom.GetAtomicNum() in [2,10,13,18]: return False
            if 21 <= atom.GetAtomicNum() <= 32: return False
            if 36 <= atom.GetAtomicNum() <= 52: return False
            if atom.GetAtomicNum() >= 54: return False
        return True
    try: std_mol = standardise.run(inp)
    except standardise.StandardiseException: return None
    if not std_mol or checkHm(std_mol) == False or checkC(std_mol) == False: return None
    else: return std_mol

#preprocess exception to catch
class MolFromSmilesError(Exception):
    'raise due to "None" from Chem.MolFromSmiles'

#preprocess exception to catch
class PreprocessViolation(Exception):
    'raise due to preprocess violation'

#calculate 2048bit morgan fingerprints, radius 2, for smiles or sdf input
def calcFingerprints(input,qtype='smiles'):
    if qtype == 'smiles': m = Chem.MolFromSmiles(input)
    else: m = input
    if not m: raise MolFromSmilesError('None mol in function')
    m = preprocessMolecule(m)
    if not m: raise PreprocessViolation('Molecule preprocessing violation')
    fp = AllChem.GetMorganFingerprintAsBitVect(m,2, nBits=2048)
    binary = fp.ToBitString()
    if qtype == 'sdf': return Chem.MolToSmiles(m), map(int,list(binary)), fp
    else: return map(int,list(binary)), fp

#calculate fingerprints for chunked array of smiles
def arrayFP(inp):
    idx, i = inp
    i = i.split(" ")
    try:
        fp, mol = calcFingerprints(i[0])
        outfp = fp
        outmol = mol
        try: outsmi_id = i[1]
        except IndexError:
            outsmi_id = i[0]
    except PreprocessViolation:
        outfp = None
        outmol = None
        outsmi_id = None
    except MolFromSmilesError:
        outfp = None
        outmol = None
        outsmi_id = None
    return outfp, outmol, outsmi_id

#import user smiles query
@logged
def importQuerySmiles(pdg, inchikey_inchi):
    conv = Converter()
    query = []
    for ik, inch in inchikey_inchi.iteritems():
        try:
            query += ["%s %s" % (conv.inchi_to_smiles(inch), ik)]
        except:
            query += ["%s %s" % ("NA", ik)]
    query = zip(range(len(query)), query)
    matrix = []
    processed_mol = []
    processed_id = []
    percent0 = 0
    if pdg.ncores == 1:
        for i, inp in enumerate(query):
            percent = int((float(i)/float(len(query)))*10)
            if percent != percent0:
                importQuerySmiles._log.debug('Processing molecules: %d%%' % (percent*10))
            percent0 = percent
            result = arrayFP(inp)
            if result[0]:
                matrix.append(result[0])
                processed_mol.append(result[1])
                processed_id.append(result[2])
    else:
        chunksize = max(1, int(len(query) / (pdg.ncores)))
        pool = Pool(processes=pdg.ncores)  # set up resources
        jobs = pool.imap(arrayFP, query, chunksize)
        for i, result in enumerate(jobs):
            percent = int((float(i)/float(len(query)))*10)
            if percent != percent0:
                importQuerySmiles._log.debug('Processing molecules: %d%%' % (percent*10))
            percent0 = percent
            if result[0]:
                matrix.append(result[0])
                processed_mol.append(result[1])
                processed_id.append(result[2])
        pool.close()
        pool.join()
    importQuerySmiles._log.debug('Processing molecules: 100%')
    matrix = np.array(matrix)
    return matrix, processed_mol, processed_id

#unzip a pkl model
def open_Model(mod_dir, mod, ntrees):
    with bz2.BZ2File(mod_dir + '/pkls' + sep + mod + '.pkl.bz2', 'rb') as bzfile:
        clf = cPickle.load(bzfile)
        #if set, increase number of trees in forest
        if ntrees and clf.n_estimators < ntrees:
            clf.set_params(n_estimators=ntrees)
    return clf

#import the training data similarity, bias and standard deviation file for given model
def getAdData(mod_dir, model_name):
    actual_mid = model_name.split('/')[-1].split('.pkl.bz2')[0]
    ad_file = mod_dir + '/ad_analysis' + sep + actual_mid + '.pkl.bz2'
    with bz2.BZ2File(ad_file, 'rb') as bzfile:
        ad_data = cPickle.load(bzfile)
    return ad_data

#perform AD analysis using full similarity weighted threshold [similarity/(bias * std_dev]
#adapted from DOI:10.1186/s13321-016-0182-y
def doSimilarityWeightedAdAnalysis(rdkit_mols, mod_dir, model_name, known, ad):
    ad_idx = []
    known = []
    ad_data = getAdData(mod_dir, model_name)
    required_threshold = np.percentile(ad_data[:,5], ad)
    for mol_idx, m in enumerate(rdkit_mols):
        ad_flag = False
        if known: k_flag = False
        else: k_flag = True
        for training_instance in ad_data:
            sim = DataStructs.TanimotoSimilarity(m,training_instance[0])
            if sim == 1.0 and k_flag == False:
                known.append([mol_idx,training_instance[1]])
                k_flag = True
            weight = sim/(training_instance[2]*training_instance[3])
            if weight >= required_threshold and ad_flag != True:
                ad_idx.append(mol_idx)
                ad_flag = True
            #if compound is in AD and no need to check accross all comps for known then break
            if k_flag == True and ad_flag == True: break
    return ad_idx, np.array(known)

#return percentile AD analysis for [similarity/(bias * std_dev] vs training data
def doPercentileCalculation(inp):
    model_name, rdkit_mols, ad_data, mod_dir = inp
    ad_data = getAdData(mod_dir, model_name)
    def calcPercentile(rdkit_mol):
        sims = DataStructs.BulkTanimotoSimilarity(rdkit_mol,ad_data[:,0])
        bias = ad_data[:,2].astype(float)
        std_dev = ad_data[:,3].astype(float)
        scores = ad_data[:,5].astype(float)
        weights = sims / (bias * std_dev)
        critical_weight = weights.max()
        percentile = percentileofscore(scores,critical_weight)
        result = percentile, None
        return result
    ret = [calcPercentile(x) for x in rdkit_mols]
    return model_name, ret

@logged
def performPercentileCalculation(pdg, models, rdkit_mols):
    # If not percentile calculation, return nans
    if not pdg.percentile:
        empty_array = np.full(len(rdkit_mols), np.nan)
        empty_array = zip(empty_array, empty_array)
        percentile_results = []
        for model in models:
            percentile_results += [(model, empty_array)]
        return percentile_results
    # Otherwise, do the percentile calculations
    performPercentileCalculation._log.debug('Starting percentile calculation...')
    input_len = len(models)
    percentile_results = np.empty(input_len, dtype=object)
    inputs = [(model, rdkit_mols, 0, pdg.mod_dir) for model in models] # The 0 accounts for the percentile ad, otherwhise we put pdg.ad
    percent0 = 0
    if pdg.ncores == 1:
        for i, inp in enumerate(inputs):
            percent = int((float(i)/float(input_len))*100)
            if percent != percent0:
                performPercentileCalculation._log.debug('Performing percentile calculation: %d%%' % (percent))
            percent0 = percent
            result = doPercentileCalculation(inp)
            if result is not None: percentile_results[i] = result
    else:
        pool = Pool(processes=pdg.ncores)
        chunksize = max(1, int(input_len / (10 * pdg.ncores)))
        jobs = pool.imap_unordered(doPercentileCalculation, inputs, chunksize)
        percent0 = 0
        for i, result in enumerate(jobs):
            percent = int((float(i)/float(input_len))*10)
            if percent != percent0:
                performPercentileCalculation._log.debug('Performing percentile calculation: %d%%' % (percent*10))
            percent0 = percent
            if result is not None: percentile_results[i] = result
        pool.close()
        pool.join()
    performPercentileCalculation._log.debug('Performing percentile calculation: 100%')
    performPercentileCalculation._log.debug('Percentile calculation completed!')
    percentile_results = [(mid, [cresult[0] for cresult in mresult]) for (mid, mresult) in percentile_results]
    return dict((x[0], x[1]) for x in percentile_results)        

#calculate standard deviation for an input compound
def getStdDev(clf, querymatrix):
    std_dev = []
    for tree in range(len(clf.estimators_)):
        std_dev.append(clf.estimators_[tree].predict_proba(querymatrix)[:,1])
    std_dev = np.clip(np.std(std_dev,axis=0),0.001,None)
    return std_dev

#raw or binary prediction worker
def doTargetPrediction(inp):
    model_name, rdkit_mols, mod_dir, ad, std, querymatrix, ntrees, known = inp
    try:
        #percentile ad analysis calculated from [near_neighbor_similarity/(bias*deviation)]
        clf = open_Model(mod_dir, model_name, ntrees)
        ret = np.zeros(len(querymatrix))
        ret.fill(np.nan)
        try:
            ad_idx, known = doSimilarityWeightedAdAnalysis(rdkit_mols, mod_dir, model_name, known, ad)
        except: return model_name, ret
        #if no mols in AD then return
        if len(ad_idx) == 0: return model_name, ret
        probs = clf.predict_proba(querymatrix[ad_idx])[:,1].clip(0.001,0.999)
        #return the standard deviation if turned on
        if std:
            std_dev = getStdDev(clf,querymatrix)
            ret[ad_idx] = std_dev[ad_idx]
            return model_name, ret
        #will only have known if was set on
        if len(known) > 0: probs[known[:,0]] = known[:,1]
        if threshold: ret[ad_idx] = map(int,probs > threshold)
        else: ret[ad_idx] = probs
    except IOError: return None
    return model_name, ret

#prediction runner for prediction or standard deviation calculation
@logged
def performTargetPrediction(pdg, models, rdkit_mols, querymatrix):
    performTargetPrediction._log.debug('Starting classification...')
    input_len = len(models)
    prediction_results = []
    inputs = [(model_name, rdkit_mols, pdg.mod_dir, 0, pdg.std, querymatrix, pdg.ntrees, pdg.known) for model_name in models] # The 0 accounts for the percentile, otherwise put pdg.ad
    percent0 = 0
    if pdg.ncores == 1:
        for i, inp in enumerate(inputs):
            percent = int((float(i)/float(input_len))*100)
            if percent != percent0:
                performTargetPrediction._log.debug('Performing classification on query molecules: %d%%' % (percent))
            percent0 = percent
            result = doTargetPrediction(inp)
            prediction_results.append(result)
    else:
        pool = Pool(processes=pdg.ncores, initializer=initPool, initargs=(querymatrix,pdg.proba,))
        chunksize = max(1, int(input_len / (10 * pdg.ncores)))
        jobs = pool.imap_unordered(doTargetPrediction, inputs, chunksize)
        for i, result in enumerate(jobs):
            percent = int((float(i)/float(input_len))*10)
            if percent != percent0:
                performTargetPrediction._log.debug('Performing classification on query molecules: %d%%' % (percent*10))
            percent0 = percent
            if result is not None: prediction_results.append(result)
        pool.close()
        pool.join()
    performTargetPrediction._log.debug('Performing classification on query molecules: 100%')
    performTargetPrediction._log.debug('Classification completed!')
    return dict((x[0], x[1]) for x in prediction_results)

#write out results
def assembleResults(results_prediction, results_percentile, query_id, proba, ad):
    mids = set(results_percentile.keys()).intersection(results_prediction.keys())
    results = collections.defaultdict(list)
    for mid in mids:
        preds = results_prediction[mid]
        percs = results_percentile[mid]
        for qy, pd, pc in zip(query_id, preds, percs):
            if proba is not None and pd < proba:
                results[qy] += []
                continue
            if ad > pc:
                results[qy] += []
                continue
            results[qy] += [(mid, round(pd,1), int(pc))]
    return results
    
#nt (Windows) compatibility initializer for the pool
def initPool(querymatrix_, threshold_=None):
    global querymatrix, threshold
    querymatrix = querymatrix_
    threshold = threshold_

# Main class

@logged
class Pidgin:

    def __init__(self, pidgin_dir = None, ncores = 1, bioactivity = [100, 10, 1, 0.1], percentile = True, proba = None, ad = 0, ortho = True, organism = None, targetclass = None, minsize = 10, p_filt = None, ntrees = None):
        """Initialize the PIDGIN v3 predictor.

        Args:
            pidgin_dir(srt): Path to the PIDGINv3 directory where models are stored.
            ncores(str): Number of cores, (default: 1).
            bioactivity(list or float): Bioactivity thresholds to use. Valid values are [100, 10, 1, 0.1] (all by default).
            percentile(bool): Perform calculation of percentiles for the applicability domain, (default = True).
            proba(float): RF probability threshold (default: None)
            ad(int): Applicability Domain (AD) filter using percentile of weights. Default: 0 (integer for percentile)'
            ortho(str): Set to use orthologue bioactivity data in model generation.
            organism(list or str): Organism filter (multiple can be specified)
            targetclass(str): Target classification filter.
            p_filt(tuple): Performance filter (validation_set[tsscv,l50so,l50po],metric[bedroc,roc,prauc,brier],performance_threshold[float]). Eg. (tsscv, bedroc, 0.5)
            ntrees(int): Minimum number of trees for warm-start random forest models (N.B Potential large latency/memory cost)
        """
        if not pidgin_dir:
            conf = Config()
            self.pidgin_dir = os.path.abspath(conf.TOOLS.pidgin_dir)
        else:
            self.pidgin_dir = os.path.abspath(pidgin_dir)
        self.ncores = ncores
        os.environ['OMP_NUM_THREADS'] = str(self.ncores)
        if type(bioactivity) is list:
            self.bioactivity = bioactivity
        else:
            self.bioactivity = [bioactivity]
        self.percentile = percentile
        assert len(set(self.bioactivity)) == len(self.bioactivity), "Repeated bioactivity values..."
        assert len(set(self.bioactivity)) == len(set(self.bioactivity).intersection([100, 10, 1, 0.1])), "Bioactivities contain values different than 100, 10, 1, 0.1"
        self.proba = proba
        assert ad >= 0 and ad <= 100, "Percentile weight not integer between 0-100%"
        self.ad = ad
        self.ortho = ortho
        if not organism:
            self.organism = organism
        else:
            if type(organism) is list:
                self.organism = organism
            else:
                self.organism = [organism]
        self.targetclass = targetclass
        self.minsize = minsize
        self.p_filt = p_filt
        self.ntrees = ntrees
        # Fixed configurations
        self.known = False
        self.se_filt = False
        self.std = False
        # Set configurations
        if self.ortho:
            self.mod_dir = os.path.join(self.pidgin_dir, "ortho")
        else:
            self.mod_dir = os.path.join(self.pidgin_dir, "no_ortho")
        
    def predict(self, inchikey_inchi):
        """Make target predictions. Returns a dictionary of the applicability domain predicted for every protein.
        
        Args:
            inchikey_inchi: A dictionary containing the molecules to be predicted.
        """
        self.__log.info('*** RUNNING PIDGIN PREDICTIONS ***')
        self.__log.info('Using ' + str(self.ncores) + ' core(s)')
        if self.ntrees: self.__log.warning('Latency warning: Number of trees will be increased to minimum: ' + str(self.ntrees))
        self.__log.info('Using bioactivity thresholds (IC50/EC50/Ki/Kd) of: ' + ",".join([str(x) for x in self.bioactivity]))
        self.__log.info('Using orthologues: ' + str(self.ortho))
        if self.organism: self.__log.info('Organism filter: ' + ",".join(self.organism))
        if self.targetclass: self.__log.info('Target class filter: ' + self.targetclass)
        if self.minsize: self.__log.info('Minimum number of actives in training: ' + str(self.minsize))
        if self.se_filt: self.__log.info('Filtering out models with Sphere Exclusion (SE)')
        #gather the models required and their information
        mid_uniprots, model_info, target_count = get_Models(self)
        self.__log.info('Total number of protein targets: ' + str(target_count))
        self.__log.info('Total number of distinct models: ' + str(len(mid_uniprots)))
        self.__log.info('Using p(activity) threshold of: ' + str(self.proba))
        self.__log.info('Importing query (calculating ECFP_4 fingerprints)')
        #import user query files
        querymatrix, rdkit_mols, query_id = importQuerySmiles(self, inchikey_inchi)
        self.__log.info('Total number of query molecules: ' + str(len(querymatrix)))
        #perform target prediction on (filtered) models (using model ids)
        results_percentile = performPercentileCalculation(self, mid_uniprots.keys(), rdkit_mols)
        results_prediction = performTargetPrediction(self, mid_uniprots.keys(), rdkit_mols, querymatrix)
        #assemble output
        self.__log.info("Assembling results")
        results_ = assembleResults(results_prediction, results_percentile, query_id, self.proba, self.ad)
        results = collections.defaultdict(list)
        for ik in inchikey_inchi.keys():
            if ik not in results_:
                results[ik] = None
            else:
                results[ik] = results_[ik]
        return dict((k,v) for k,v in results.iteritems())
