import os
import sys
#sys.path.append('/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/')
import json
import time
import logging
import collections
import numpy as np

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset
from chemicalchecker.util import logged

cutoff_idx = 5  # what we consider similar (dist < p-value 0.02)
best = 20  # top molecules in libraries
dummy = 999  # distance bin for non-similar
overwrite = True  # we will overwrite existing json files
max_neig = 500  # we only take top N neighbours foreach space (memory reason)

def _check_keys_presence_on_spaces( cc ):
    metric_obs = None
    metric_prd = None
    map_coords_obs = collections.defaultdict(list)
    dataset_pairs = {}
    for ds in Dataset.get(exemplary=True):
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
    return metric_obs, metric_prd, map_coords_obs, dataset_pairs
    
def _compute_dist_cutoffs( cc, dataset_pairs, metric_obs, metric_prd ):
    dss = { 'obs': 'sign1', 'prd': 'sign3' }
    bg_vals = dict()
    signatures = dict()
    for k in dss:
        bg_vals[k] = {}
        signatures[k] = {}
    for coord in dataset_pairs.keys():
        for k in dss:
            cctype = dss[k]
            sign = cc.get_signature( cctype, "reference", dataset_pairs[coord])
            bg_vals[k][coord] = sign.background_distances( eval( f"metric_{k}" ) )["distance"]
            signatures[k][coord] = sign
    return bg_vals, signatures     

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
    
def _get_matrix_value( info, dataset, ik, dist, flag_isInCurrent ):
    square, type_data = dataset.split("_")
    pos, sign, pts = index_sign(dataset)
    
    M = info['matrix']
    iksm = set( info['ink_pos_mat'] )
    if( not ik in iksm ):
        arr = [np.nan] * 26
        iksm.add(ik)
        info['ink_pos_mat'][ik] = len(iksm)
        M = np.vstack([ M, arr ])    
        
    t = info['ink_pos_mat'][ik]
    val = M[t, pos]
    # if generic neighbor doesn't have a value set it
    # if it is in current space, update points matrix
    if( flag_isInCurrent ):
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
    info['matrix'] = M
    
    return info

@logged(logging.getLogger("[ WEB-SIMILARS ]"))
def main():

    cc = ChemicalChecker('/aloy/web_checker/package_2024_update/')

    # read mapping inchi names
    names_jason = '/aloy/scratch/ymartins/pipelines/cc_update_2024_02/tmp/inchies_names.json'
    with open(names_jason) as json_data:
        inchies_names = json.load(json_data)
    inchikeys = list(inchies_names.keys())[:100]
    inchikeys = ['MXKSDRXNOXZPDY-AMHTUMDSSA-N'] # gold standard json

    neig = cc.get_signature( 'neig1', "full", 'B1.001' )
    d = neig.get_vectors_multiple_datasets( inchikeys, True, ['distances', 'indices'] )
       
    metric_obs, metric_prd, map_coords_obs, dataset_pairs = _check_keys_presence_on_spaces(cc)
    bg_vals, signatures = _compute_dist_cutoffs( cc, dataset_pairs, metric_obs, metric_prd )
    version = '2024_02'
    
    # read library bioactive
    lib_bio_file = '/aloy/scratch/ymartins/pipelines/cc_update_2024_02/tmp/lib_bio.json'
    with open(lib_bio_file) as json_data:
        ref_bioactive = json.load(json_data)
        libs = set(ref_bioactive.keys())
        libs.add("All Bioactive Molecules")

    matrix = np.full( ( 1, 26 ), np.nan )
    temp_info = {}
    for index, inchikey in enumerate(inchikeys):
        temp_info[inchikey] = { 'selected': set(), 'matrix': matrix, 'ink_pos_mat': {}, "ref_counts": {}, 'inchies': {}, 'selected': list(), "all_neig": list(), 'inks_by_ds': {}  }
        temp_info[inchikey]['matrix'][:, 25] = 0
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


    # ------------------------------ Dataset area
    t00 = time.time()
    dc=0
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
        nn_dist = d['distances']['signs']
        nn_inks = d['indices']['signs']
        
        tempinchikeys = dat['distances']['inks'].tolist()
        print( nn_dist.shape, nn_inks.shape, len( tempinchikeys ), len( set( tempinchikeys ).intersection( set( inchikeys ) ) ) )
        #_, nn_dist = neig.get_vectors( inchikeys, include_nan=True, dataset_name='distances')
        #_, nn_inks = neig.get_vectors( inchikeys, include_nan=True, dataset_name='indices')
       
        # mask to keep only neighbors below cutoff
        masks = nn_dist <= dist_cutoffs[cutoff_idx]
        # get binned data according to distance cutoffs
        dist_bin = np.digitize(nn_dist, dist_cutoffs)
        # get close neighbors inchikeys and distance bins and apply mapping
        mappings = signatures[type_data][coord].get_h5_dataset('mappings')
        ref_mapping = list(mappings[:, 1])
        all_inks = list()
        all_dbins = list()
        
        t0 = time.time()
        # couldn't find a way to avoid iterating on molecules
        index_ik = 0
        for ref_nn_ink, ref_dbin, mask in zip(nn_inks, dist_bin, masks):
            inchikey = tempinchikeys[index_ik]

            # apply distance cutoff
            ref_nn_ink = ref_nn_ink[mask]
            ref_dbin = ref_dbin[mask]
            ref_dbin_mapping = dict( zip(ref_nn_ink, ref_dbin) )
            
            # get idx bassed on redundant 'reference' column
            full_idxs = np.isin(ref_mapping, list(ref_nn_ink) )
            
            # get ref to full mapping (redundat mols as lists)
            full_ref_mapping = dict( mappings[full_idxs] )
            #ref_full_mapping = collections.defaultdict(list)
            
            aux = set()
            
            distances = {}
            for k, v in full_ref_mapping.items():
                #if( len(aux) < max_neig ):
                aux.add(k)
                dist = ref_dbin_mapping[v]
                
                distances[k] = dist
                # if( type_data == 'prd' or ( coord in map_coords_obs[inchikey] and type_data == 'obs' ) ): ------ Use later
                temp_info[inchikey] = _get_matrix_value( temp_info[inchikey], dataset, k, dist, True )
                
                
                for lib in libs:
                    # if we already selected enought stop
                    if temp_info[inchikey]['ref_counts'][lib][pos] >= best:
                        break
                    if lib == 'All Bioactive Molecules':
                        found = True
                    else:
                        found = k in ref_bioactive[lib]
                    if found:
                        temp_info[inchikey]['ref_counts'][lib][pos] += 1
                        temp_info[inchikey]['selected'].append(k)
                        temp_info[inchikey]['inchies'][lib].append(k)
                                
                #else:
                #    break
                    
            temp_info[inchikey]['all_neig'].extend( list(aux) )
            if( len(aux) > 0 ):
                temp_info[inchikey]['inks_by_ds'][dataset] = distances
            
            index_ik += 1
            
            print(index_ik, inchikey, time.time()-t0)
            #main._log.info( f"{index_ik} - {inchikey} key: {time.time()-t0}" )
            
            print(temp_info[inchikey])
            
            """
                ref_full_mapping[v].append(k)
            # aggregate mappings
            full_inks = [ref_full_mapping[i] for i in ref_nn_ink]
            full_inks = [item for sublist in full_inks for item in sublist]
            full_dbins = [[ref_dbin_mapping[i]] * len(ref_full_mapping[i]) for i in ref_nn_ink]
            full_dbins = [item for sublist in full_dbins for item in sublist]
            
            all_inks.append(full_inks[:max_neig])
            all_dbins.append(full_dbins[:max_neig])
            """
        print(dataset, ' dataset took %.3f secs', time.time() - t0)
        main._log.info( f' {dataset} dataset took { time.time() - t0 } secs')
        
        dc += 1
    print( ' first part took %.3f secs', time.time() - t00)
    main._log.info( f'first part took { time.time() - t0 } secs')
    
        
    # ------------------ new loop
    save_file_path = './'
    task_id = 1
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
        
    t0 = time.time()
    for index, inchikey in enumerate(inchikeys):
        keys = set( temp_info[inchikey]['inks_by_ds'] )
        
        for dataset in keys:
            coord, type_data = dataset.split("_")
            neigs_to_update = set(temp_info[inchikey]['all_neig']) - set(temp_info[inchikey]['inks_by_ds'][dataset])
            for n in neigs_to_update:
                if( type_data == 'prd' or ( coord in map_coords_obs[n] and type_data == 'obs' ) ):
                    dist = temp_info[inchikey]['inks_by_ds'][dataset][n]
                    temp_info[inchikey]['matrix'] = _get_matrix_value( temp_info[inchikey], dataset, n, dist, False )
            
        # convert to lists
        selected = temp_info[inchikey]['selected']
        selected = list(selected)
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
        with open( f'gold_{inchikeys[0]}_tempinchi.json', 'w') as g:
            json.dump(inchies, g)
        
        tempfile.write(f"{ inchikey }\t{ version }\t{ jsontxt }\n".encode('UTF-8') )
     
    tempfile.write("\\.\n".encode('UTF-8') )
    tempfile.close() 
    print( ' second part took %.3f secs', time.time() - t0)
    main._log.info( f'second part took { time.time() - t0 } secs')
    
main()    
    
"""    
# keep neighbors and bins for later
ds_inks_bin[dataset] = (all_inks, all_dbins)



# ---------------------- Avoid need of this part

for index, inchikey in enumerate(inchikeys):
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
    for dataset in keys:
        square, type_data = dataset.split("_")
        pos, sign, pts = index_sign(dataset)

        # iterate on all generic neigbors
        for t, ik in enumerate(all_neig):
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

        # select top neighbors in current space that are also part of
        # libraries
        for ik in neig_ds[dataset]:
            # never select self
            if ik == inchikey:
                continue
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
            inchies[sel]["name"] = inchies_names[sel]
        else:
            inchies[sel]["name"] = ""
     """       
