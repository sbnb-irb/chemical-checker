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
from tqdm import tqdm

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

def _get_matrix_value_batch( info, dataset, iks, dists, flag_isInCurrent, map_coords_obs ):
    coord, type_data = dataset.split("_")
    pos, sign, pts = index_sign(dataset)
    
    M = info['matrix']
    
    def _search_coord(ik):
        return ( type_data=='prd' or ( type_data=='obs' and coord in map_coords_obs[ik] ) )
    
    iks = list(iks)
    all_keys = np.array( list(info['ink_pos_mat']) )
    all_idx = np.array( list(info['ink_pos_mat'].values()) )
    bin_idxs = _is_in_set_nb( all_keys, np.array(iks) )
    filt_keys = all_keys[ bin_idxs ]
    filt_idx = all_idx[ bin_idxs ]
    aux = dict( zip(filt_keys, filt_idx) )
    
    def _check_prior_val(ik):
        return ( not( M[ aux[ik], pos] > 0 ) )
    bin_idx0 = np.vectorize(_check_prior_val)( iks )
    bin_idxs = np.array( [True]*len(filt_idx) )
    bin_idxs = bin_idxs & bin_idx0
    
    if( np.all( bin_idxs == False ) ):
        return info
    else:
        val = dists
        # otherwise check if we can say they are different
        if(not flag_isInCurrent):
            bin_idx2 = np.vectorize(_search_coord)(iks)
            bin_idxs = bin_idxs & bin_idx2
            val = dummy
        else:
            temp = dict( zip(iks, dists.tolist() ) )
            new_keys = filt_keys[ bin_idxs ]
            dists = list( map( lambda x: temp[x], new_keys))
            val = np.array( dists )
            
        t = filt_idx[ bin_idxs ]
        M[t, pos] = sign * val
        
        if( flag_isInCurrent ):
            M[t, 25] += pts
        
        info['matrix'] = M
    
    return info

def _load_presence_libs(universe, ref_bioactive):
    u = set(universe)
    b = [True]*len(u)
    all_ = dict( zip(list(u), b) )
    dc = { 'All Bioactive Molecules': all_ }
    for l in ref_bioactive:
        ok = u.intersection( ref_bioactive[l] )
        nok = u.difference( ref_bioactive[l] )
        a = list(ok)
        a.extend( list(nok) )
        b = ( [True]*len(ok) )
        c = ( [False]*len(nok) )
        b.extend( c )
        
        dc[l] = dict( zip( a,b ) )
        
    return dc

def _is_in_set_nb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

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
    
    # read inchikey to pubmed names mapping
    with open(names_jason) as json_data:
        inchies_names = json.load(json_data)

    # read library bioactive
    with open(lib_bio_file) as json_data:
        ref_bioactive = json.load(json_data)
    libs = set(ref_bioactive.keys())
    libs.add("All Bioactive Molecules")
    
    universe = set(map_coords_obs)
    found = _load_presence_libs(universe, ref_bioactive)
    
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
    
    matrix = np.full( ( 0, 26 ), np.nan )
    temp_info = {}
    for index, inchikey in enumerate(inchikeys):
        temp_info[inchikey] = { 'selected': list(), 'matrix': [], 'ink_pos_mat': {}, "ref_counts": {}, 'inchies': {}, "all_neig": set(), 'inks_by_ds': {}  }
        #temp_info[inchikey]['matrix'][:, 25] = 0
        for lib in libs:
            temp_info[inchikey]['ref_counts'][lib] = [0] * 25
            temp_info[inchikey]['inchies'][lib] = list()
    
    keys = [k + "_obs" for k in dataset_pairs.keys()] + \
            [k + "_prd" for k in dataset_pairs.keys()]
    ds_inks_bin = {}
    neig_cctype = {
        'obs': 'neig1',
        'prd': 'neig3',
    }
    
    t00 = time.time()
    dc=0
    tempdata = {}
    #keys = ['A3_obs']
    for dataset in keys:
        pos, sign, pts = index_sign(dataset)
        
        print( 'processing dataset ', dataset )
        t0 = time.time()
        #dataset = keys[1]
        coord, type_data = dataset.split("_")
        dist_cutoffs = bg_vals[type_data][coord]

        neig = cc.get_signature( neig_cctype[type_data], "full", dataset_pairs[coord])
        dat = neig.get_vectors_multiple_datasets( inchikeys, True, ['distances', 'indices'], max_neig )
        nn_dist = dat['distances']['signs']
        nn_inks = dat['indices']['signs']
        
        tempinchikeys = dat['distances']['inks'].tolist()
        
        #_, nn_dist = neig.get_vectors( inchikeys, include_nan=True, dataset_name='distances')
        #_, nn_inks = neig.get_vectors( inchikeys, include_nan=True, dataset_name='indices')
       
        # mask to keep only neighbors below cutoff
        masks = nn_dist <= dist_cutoffs[cutoff_idx]
        # get binned data according to distance cutoffs
        dist_bin = np.digitize(nn_dist, dist_cutoffs)
        # get close neighbors inchikeys and distance bins and apply mapping
        mappings = signatures[type_data][coord].get_h5_dataset('mappings')
        ref_mapping = list(mappings[:, 1])
        
        #t0 = time.time()
        # couldn't find a way to avoid iterating on molecules
        index_ik = 0
        for ref_nn_ink, ref_dbin, mask in zip(nn_inks, dist_bin, masks):
            inchikey = tempinchikeys[index_ik]

            # apply distance cutoff
            ref_nn_ink = ref_nn_ink[mask]
            ref_dbin = ref_dbin[mask].tolist()
            ref_dbin_mapping = dict( zip(ref_nn_ink, ref_dbin) )
            
            # get idx bassed on redundant 'reference' column
            #full_idxs = np.isin(ref_mapping, list(ref_nn_ink) )
			full_idxs = np.isin(ref_mapping, list(ref_nn_ink))
			# get ref to full mapping (redundat mols as lists)
			full_ref_mapping = dict(mappings[full_idxs])
			
			ref_full_mapping = collections.defaultdict(list)
			for k, v in full_ref_mapping.items():
				ref_full_mapping[v].append(k)
			# aggregate mappings
			full_inks = [ref_full_mapping[i] for i in ref_nn_ink]
			full_inks = [item for sublist in full_inks for item in sublist]
			full_dbins = [[ref_dbin_mapping[i]] *
						  len(ref_full_mapping[i]) for i in ref_nn_ink]
			full_dbins = [item for sublist in full_dbins for item in sublist]
            
            temp_info[inchikey]['all_neig'].update( full_inks )
            distances = dict( zip( full_inks, full_dbins ) )
            if( len(ks) > 0 ):
                temp_info[inchikey]['inks_by_ds'][dataset] = distances
                
            filtered = mappings[full_idxs]
            ks = filtered[:max_neig, 0]
            dist = list( map( lambda x: ref_dbin_mapping[x], filtered[:max_neig, 1] ) )
            distances = dict( zip( ks, dist ) )
            temp_info[inchikey]['all_neig'].update( list(ks) )
            if( len(ks) > 0 ):
                temp_info[inchikey]['inks_by_ds'][dataset] = distances
            
            index_ik += 1
            
            if( index_ik % 50 == 0 ):
                #print( f'Last 50: {index_ik} {time.time()-t0}' )
                main._log.info( f'Last 50: {index_ik} {time.time()-t0}' )
                main._log.info("  MEM USED: {:>5.1f} GB "
                               "(\u0394 {:>5.3f} GB)".format(*mem()))
            #main._log.info( f"{index_ik} - {inchikey} key: {time.time()-t0}" )
            
            #print(temp_info[inchikey])
        #print(dataset, ' dataset took %.3f secs', time.time() - t0)
        main._log.info( f' {dataset} dataset took { time.time() - t0 } secs')
        
        main._log.info("  MEM USED: {:>5.1f} GB "
                               "(\u0394 {:>5.3f} GB)".format(*mem()))
        
        dc += 1
    #print( ' first part took %.3f secs', time.time() - t00)
    main._log.info( f'first part took { time.time() - t00 } secs')
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    
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
    
    t00 = time.time()
    for index, inchikey in tqdm( enumerate(inchikeys) ):
        t0 = time.time()
        
        all_neigh = set( temp_info[inchikey]['all_neig'] )
        all_neig = np.array( sorted( list(all_neigh) ) )
        temp_info[inchikey]['ink_pos_mat'] = dict( zip( all_neig, np.arange(len(all_neig)) ) )
        temp_info[inchikey]['matrix'] = np.full( ( len(all_neig), 26 ), np.nan )
        temp_info[inchikey]['matrix'][:, 25] = 0
        
        temp_info[inchikey]['selected'] = set()
        
        keys = set( temp_info[inchikey]['inks_by_ds'] )
        print(inchikey, 'all neig length', len(all_neig) )
        
        t1 = time.time()
        for dataset in keys:
            coord, type_data = dataset.split("_")
            pos, sign, pts = index_sign(dataset)
            
            in_dataset = set(temp_info[inchikey]['inks_by_ds'][dataset])
            dists = np.array( list( map( lambda x: temp_info[inchikey]['inks_by_ds'][dataset][x], in_dataset ) ) )
            temp_info[inchikey] = _get_matrix_value_batch( temp_info[inchikey], dataset, in_dataset, dists, True, map_coords_obs )
            out_dataset = all_neigh - in_dataset
            temp_info[inchikey] = _get_matrix_value_batch( temp_info[inchikey], dataset, out_dataset, None, False, map_coords_obs )
            
            aux = np.array( list( in_dataset ) )
            for lib in libs:
                
                if temp_info[inchikey]['ref_counts'][lib][pos] >= best:
                    break
                    
                def _check_lib(ik):
                    return found[lib][ik]
                    
                bin_lib = np.vectorize( _check_lib )( aux )
                max_ = best - temp_info[inchikey]['ref_counts'][lib][pos]
                filtered = aux[ bin_lib ]
				ok_lib = set( filtered ).difference( set( temp_info[inchikey]['inchies'][lib] ) )
				idxs = np.isin( filtered, list(ok_lib) )
				ok_lib = filtered[idxs][:max_]
                
                temp_info[inchikey]['ref_counts'][lib][pos] += len(ok_lib)
                temp_info[inchikey]['inchies'][lib].extend( ok_lib )
                temp_info[inchikey]['selected'].update( ok_lib )
                 
        #print( ' iterating over all neig took %.3f secs', time.time() - t1)
        #main._log.info( f' iterating over all neig took { time.time() - t1 } secs')        
           
        max_selected = best * len(keys) * len(libs) 
        # convert to lists
        selected = temp_info[inchikey]['selected']
        selected = list(selected)[:max_selected]
        for lib in libs:
            temp_info[inchikey]['inchies'][lib] = list( temp_info[inchikey]['inchies'][lib])
        
        M = temp_info[inchikey]['matrix']
        ink_pos = temp_info[inchikey]['ink_pos_mat']
        
        inchies = temp_info[inchikey]['inchies']
        # save neigbors data for explore page
        for sel in selected:
            inchies[sel] = {}
            inchies[sel]["inchikey"] = sel
            inchies[sel]["data"] = [None if np.isnan(x) else x for x in M[ ink_pos[sel] ] ]
            if sel in inchies_names:
                inchies[sel]["name"] = inchies_names[sel].replace('\\', '').replace('"', '').replace("'", '')
            else:
                inchies[sel]["name"] = ""
        
        jsontxt = json.dumps(inchies)
        tempfile.write(f"{ inchikey }\t{ version }\t{ jsontxt }\n".encode('UTF-8') )
		
		#js = json.dumps( inchies )
		upq = f"update similars set explore_data='{ jsontxt }' where inchikey = '{ inchikey }'"
		res = psql.query(upq, dbname )
        
        #print( inchikey, ' ik took %.3f secs', time.time() - t0)
        #main._log.info( f' {inchikey} ik took { time.time() - t0 } secs')
        
        if(index % 50 == 0):
            main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
        
        del temp_info[inchikey]
     
    tempfile.write("\\.\n".encode('UTF-8') )
    tempfile.close() 
    #print( ' second part took %.3f secs', time.time() - t00)
    main._log.info( f'second part took { time.time() - t00 } secs')
    main._log.info("MEM USED: {:>5.1f} GB (\u0394 {:>5.3f} GB)".format(*mem()))
    
    c = Config()
    host = c.DB.host
    user = c.DB.user
    passwd = c.DB.password
    table_new = 'similars'
    db_new = dbname
    
    #_restore_similar_data_from_chunks( host, db_new, user, passwd, PATH)
    
if __name__ == '__main__':
    main(sys.argv[1:])

