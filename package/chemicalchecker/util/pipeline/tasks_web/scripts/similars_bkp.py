import os
import sys
import time
import json
import pickle
import psutil
import logging
import argparse
import collections
import numpy as np

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset
from chemicalchecker.util import logged
from chemicalchecker.util import Config

cutoff_idx = 5  # what we consider similar (dist < p-value 0.02)
best = 20  # top molecules in libraries
dummy = 999  # distance bin for non-similar
overwrite = True  # we will overwrite existing json files
max_neig = 1000  # we only take top N neighbours foreach space (memory reason)


class MemMonitor:

    def __init__(self):
        self.last = 0

    def memused(self):
        """Returns memory used in GB."""
        return psutil.Process().memory_info().rss / (1024 * 1024 * 1024)

    def __call__(self):
        curr = self.memused()
        inc = curr - self.last
        self.last = curr
        return curr, inc

def _restore_similar_data_from_chunks( host_name,database_name,user_name,database_password, outfile):
    command = 'PGPASSWORD={4} psql -h {0} -d {1} -U {2} -f {3}'\
              .format(host_name, database_name, user_name, outfile, database_password)
    os.system( command )
    #os.remove( outfile )

def index_sign(dataset):
    offset = {'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20}
    if dataset.endswith('prd'):
        sign = -1
        pts = 1
    else:
        sign = 1
        pts = 2
    char, num = dataset[0], int(dataset[1])
    num -= 1
    col = offset[char] + num
    return col, sign, pts


def get_parser():
    description = 'This script will produce the json of each molecule with '
    'the information reported in the `explore` page of individual'
    'molecules. Sign1 and sign3 are used for fetching neighbors.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('task_id', type=str,
                        help='The chunk of data to work on')
    parser.add_argument('filename', type=str,
                        help='pickled file with lists of molecules')
    parser.add_argument('names_jason', type=str,
                        help='Inchi-pubmed mapping')
    parser.add_argument('lib_bio_file', type=str,
                        help='bioactive molecules lists')
    parser.add_argument('save_file_path', type=str,
                        help='Root of sql temporary directory')
    parser.add_argument('dbname', type=str,
                        help='name of the DB (not used)')
    parser.add_argument('version', type=str,
                        help='Version of the CC')
    parser.add_argument('CC_ROOT', type=str,
                        help='Root of CC instance')
    return parser


@logged(logging.getLogger("[ WEB-SIMILARS ]"))
def main(args):
    # get script arguments
    args = get_parser().parse_args(args)
    for k, v in vars(args).items():
        main._log.info("[ARGS] {:<22}: {:<}".format(k, v))
    task_id = args.task_id
    filename = args.filename
    names_jason = args.names_jason
    lib_bio_file = args.lib_bio_file
    save_file_path = args.save_file_path
    dbname = args.dbname
    version = args.version
    CC_ROOT = args.CC_ROOT
    mem = MemMonitor()

    # input is a chunk of universe inchikey
    datIn = pickle.load(open(filename, 'rb'))[task_id]
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    inchikeys = datIn['keys']
    metric_obs  = datIn['metric_obs']
    metric_prd  = datIn['metric_prd']
    map_coords_obs  = datIn['map_coords_obs']
    dataset_pairs  = datIn['dataset_pairs']
    bg_vals  = datIn['bg_vals']
    signatures = datIn['signatures']
    
    cc = ChemicalChecker( CC_ROOT )
    
    # for each molecule check if json is already available
    """
    if not overwrite:
        notdone = list()
        for index, inchikey in enumerate(inchikeys):
            PATH = save_file_path + "/%s/%s/%s/" % (
                inchikey[:2], inchikey[2:4], inchikey)
            filename = PATH + '/explore_' + version + '.json'
            if os.path.isfile(filename):
                try:
                    json.load(open(filename, 'r'))
                except Exception:
                    notdone.append(inchikey)
                    continue
            else:
                notdone.append(inchikey)
        if len(notdone) == 0:
            main._log.info('All molecules already present, nothing to do.')
            sys.exit()
        else:
            inchikeys = notdone
    """
    
    """
    # for each molecule which spaces are available in sign1?
    main._log.info('')
    main._log.info('1. Determine available spaces in sign1 for each molecule.')
    t0 = time.time()
    cc = ChemicalChecker(CC_ROOT)
    metric_obs = None
    metric_prd = None
    map_coords_obs = collections.defaultdict(list)
    dataset_pairs = {}
    for ds in Dataset.get(exemplary=True):
        main._log.info('   %s', ds)
        dataset_pairs[ds.coordinate] = ds.dataset_code
        if metric_obs is None:
            neig1 = cc.get_signature("neig1", "reference", ds.dataset_code)
            metric_obs = neig1.get_h5_dataset('metric')[0]
        if metric_prd is None:
            neig3 = cc.get_signature("neig3", "reference", ds.dataset_code)
            metric_prd = neig3.get_h5_dataset('metric')[0]
        sign1 = cc.get_signature("sign1", "full", ds.dataset_code)
        keys = sign1.unique_keys
        for ik in keys:
            map_coords_obs[ik] += [ds.coordinate]
    main._log.info('1. took %.3f secs', time.time() - t0)
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))

    # get relevant background distances and mappings
    main._log.info('')
    main._log.info('2. Pre-fetch background distances')
    t0 = time.time()
    bg_vals = dict()
    bg_vals['obs'] = dict()
    bg_vals['prd'] = dict()
    signatures = dict()
    signatures['obs'] = dict()
    signatures['prd'] = dict()
    for coord in dataset_pairs.keys():
        main._log.info('  %s', coord)
        sign1 = cc.get_signature("sign1", "reference", dataset_pairs[coord])
        bg_vals['obs'][coord] = sign1.background_distances(metric_obs)[
            "distance"]
        signatures['obs'][coord] = sign1
        sign3 = cc.get_signature("sign3", "reference", dataset_pairs[coord])
        bg_vals['prd'][coord] = sign3.background_distances(metric_prd)[
            "distance"]
        signatures['prd'][coord] = sign3
    main._log.info('2. took %.3f secs', time.time() - t0)
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    """
    # for both observed (sign1) and predicted (sign3) get significant neighbors
    main._log.info('')
    main._log.info('3. Pre-fetch significant neighbors')
    t0 = time.time()
    keys = [k + "_obs" for k in dataset_pairs.keys()] + \
        [k + "_prd" for k in dataset_pairs.keys()]
    ds_inks_bin = {}
    neig_cctype = {
        'obs': 'neig1',
        'prd': 'neig3',
    }
    for dataset in keys:
        main._log.info('  %s', dataset)
        coord, type_data = dataset.split("_")
        dist_cutoffs = bg_vals[type_data][coord]
        
        neig = cc.get_signature( neig_cctype[type_data], "full", dataset_pairs[coord])
        _, nn_dist = neig.get_vectors( inchikeys, include_nan=True, dataset_name='distances')
        _, nn_inks = neig.get_vectors( inchikeys, include_nan=True, dataset_name='indices')
        # mask to keep only neighbors below cutoff
        masks = nn_dist <= dist_cutoffs[cutoff_idx]
        # get binned data according to distance cutoffs
        dist_bin = np.digitize(nn_dist, dist_cutoffs)
        # get close neighbors inchikeys and distance bins and apply mapping
        main._log.info('  Fetched reference NN, mapping to full')
        mappings = signatures[type_data][coord].get_h5_dataset('mappings')
        ref_mapping = list(mappings[:, 1])
        all_inks = list()
        all_dbins = list()
        # couldn't find a way to avoid iterating on molecules
        c = 0
        t2 = time.time()
        t3 = time.time()
        for ref_nn_ink, ref_dbin, mask in zip(nn_inks, dist_bin, masks):
            # print progress
            if not(c % 10):
                avg = 0
                if c != 0:
                    avg = (time.time() - t3) / c
                main._log.info('  %s out of %s, took %.3f (avg/mol: %.3f s.)' %
                               (c, len(nn_inks), time.time() - t2, avg))
                main._log.info("  MEM USED: {:>5.1f} GB "
                               "(\u0394 {:>5.3f} GB)".format(*mem()))
                t2 = time.time()
            # apply distance cutoff
            ref_nn_ink = ref_nn_ink[mask]
            ref_dbin = ref_dbin[mask]
            ref_dbin_mapping = dict(zip(ref_nn_ink, ref_dbin))

            # get idx bassed on redundant 'reference' column
            full_idxs = np.isin(ref_mapping, list(ref_nn_ink))
            # get ref to full mapping (redundat mols as lists)
            full_ref_mapping = dict(mappings[full_idxs])
            ref_full_mapping = collections.defaultdict(list)
            
            aux = set()
            full_inks = []
            full_dbins = []
            for k, v in full_ref_mapping.items():
                if( len(full_dbins) < max_neig ):
                    if( not k in aux ):
                        aux.add(k)
                        full_inks.append( k )
                        full_dbins.append( ref_dbin_mapping[v] )
                else:
                    break

            """
            # this iterate on bins to aggregate mappings (removed to avoid 
            # multiple call np.isin that is slow)
            full_inks = list()
            full_dbins = list()
            unique_dbin = np.unique(ref_dbin)
            for dbin in unique_dbin:
                # get inks in the bin
                ink_dbin = ref_nn_ink[ref_dbin == dbin]
                # get idx bassed on redundant 'reference' column
                full_idxs = np.isin(ref_mapping, list(ink_dbin))
                # get non redundnt 'full' inks
                full_nn_ink = mappings[:,0][full_idxs]
                # append to molecule lists
                full_inks.extend(full_nn_ink)
                full_dbins.extend([dbin] * len(full_nn_ink))
            """
            all_inks.append(full_inks[:max_neig])
            all_dbins.append(full_dbins[:max_neig])
            c += 1

        # keep neighbors and bins for later
        ds_inks_bin[dataset] = (all_inks, all_dbins)
    main._log.info('3. took %.3f secs', time.time() - t0)
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))

    # read inchikey to pubmed names mapping
    with open(names_jason) as json_data:
        inchies_names = json.load(json_data)

    # read library bioactive
    with open(lib_bio_file) as json_data:
        ref_bioactive = json.load(json_data)
    libs = set(ref_bioactive.keys())
    libs.add("All Bioactive Molecules")
    
    PATH = os.path.join( save_file_path, f"lines_task-{task_id}.sql" )
    tempfile = open(PATH, "wb")
    tempfile.write("""
    --
    -- PostgreSQL database dump
    --

    SET statement_timeout = 0;
    SET lock_timeout = 0;
    SET client_encoding = 'UTF8';
    SET standard_conforming_strings = on;
    SELECT pg_catalog.set_config('search_path', '', false);
    SET check_function_bodies = false;
    SET xmloption = content;
    SET client_min_messages = warning;
    SET row_security = off;

    SET default_tablespace = '';

    COPY public.similars (inchikey, version, explore_data) FROM stdin;
    """.encode('UTF-8') )
    
    main._log.info('')
    main._log.info('4. Save explore json')
    t0_tot = time.time()
    # save in each molecule path the file the explore json (ranked neighbors)
    for index, inchikey in enumerate(inchikeys):
        t0 = time.time()
        # only consider spaces where the molecule is present
        keys = [k + "_obs" for k in map_coords_obs[inchikey]] + \
            [k + "_prd" for k in dataset_pairs.keys()]

        # check if there are neighbors and keep their distance bin
        all_neig = set()
        neig_ds = dict()
        empty_spaces = list()
        for dataset in keys:
            inks = ds_inks_bin[dataset][0][index]
            if len(inks) == 0:
                empty_spaces.append(dataset)
                continue
            # iterate on each neighbor and expand to full set
            all_neig.update(set(inks))
            dbins = ds_inks_bin[dataset][1][index]
            neig_ds[dataset] = dict(zip(inks, dbins))
        for ds in empty_spaces:
            keys.remove(ds)

        # join and sort all neighbors from all spaces obs and pred
        all_neig = np.array(sorted(list(all_neig)))
        ink_pos = dict(zip(all_neig, np.arange(len(all_neig))))
        M = np.full((len(all_neig), 26), np.nan)
        M[:, 25] = 0

        # keep track of neigbors in reference libraries
        inchies = dict()
        ref_counts = dict()
        for lib in libs:
            ref_counts[lib] = [0] * 25
            inchies[lib] = set()

        # rank all neighbors
        
        selected = set()
        for t, ik in enumerate(all_neig):
            # iterate on all generic neigbors
            for dataset in keys:
                square, type_data = dataset.split("_")
                pos, sign, pts = index_sign(dataset)
            
                val = M[t, pos]
                # if generic neighbor has value from obs leave it
                if val > 0:
                    continue
                # if generic neighbor doesn't have a value set it
                # if it is in current space, update points matrix
                if ik in neig_ds[dataset]:
                    dist = neig_ds[dataset][ik]
                    M[t, pos] = sign * dist
                    M[t, 25] += pts
                # otherwise check if we can say they are different
                else:
                    # if dataset is obs check against molecules in sign1
                    if type_data == 'obs':
                        if square in map_coords_obs[ik]:
                            M[t, pos] = sign * dummy
                    # if dataset is prd check against universe
                    else:
                        if ik in map_coords_obs:
                            M[t, pos] = sign * dummy

                if ( (ik in neig_ds[dataset]) and (ik != inchikey) ):
                    for lib in libs:
                        # if we already selected enought stop
                        if ref_counts[lib][pos] >= best:
                            break
                        if lib == 'All Bioactive Molecules':
                            found = True
                        else:
                            found = ik in ref_bioactive[lib]
                        if found:
                            ref_counts[lib][pos] += 1
                            selected.add(ik)
                            inchies[lib].add(ik)

        # convert to lists
        selected = list(selected)
        for lib in libs:
            inchies[lib] = list(inchies[lib])

        # save neigbors data for explore page
        for sel in selected:
            inchies[sel] = {}
            inchies[sel]["inchikey"] = sel
            inchies[sel]["data"] = [None if np.isnan(x) else x for x in M[
                ink_pos[sel]]]
            if sel in inchies_names:
                inchies[sel]["name"] = inchies_names[sel].replace('\\', '').replace('"', '').replace("'", '')
            else:
                inchies[sel]["name"] = ""
        """
        PATH = save_file_path + "/%s/%s/%s/" % (
            inchikey[:2], inchikey[2:4], inchikey)
        with open(PATH + '/explore_' + version + '.json', 'w') as outfile:
            json.dump(inchies, outfile)
        """
        jsontxt = json.dumps(inchies).replace("'","\\'")
        tempfile.write(f"{ inchikey }\t{ version }\t{ jsontxt }\n".encode('UTF-8') )
        
        main._log.info('  %s took %.3f secs', inchikey, time.time() - t0)
        main._log.info(
            "  MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    main._log.info('4. Saving all took %.3f secs', time.time() - t0_tot)
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    
    tempfile.write("\\.\n".encode('UTF-8') )
    tempfile.close()
    
    c = Config()
    host = c.DB.host
    user = c.DB.user
    passwd = c.DB.password
    table_new = 'similars'
    db_new = dbname
    
    _restore_similar_data_from_chunks( host, db_new, user, passwd, PATH)
    
if __name__ == '__main__':
    main(sys.argv[1:])

