#!/miniconda/bin/python


'''

Download and slightly process STRING.

'''

# Imports
import xml.etree.ElementTree as pxml
import os
import collections
import itertools as itt
import networkx as nx
import numpy as np
import operator
import sys
sys.path.append(os.path.join(sys.path[0],"../../src/utils"))
sys.path.append(os.path.join(sys.path[0],"../config"))
from checkerUtils import logSystem, execAndCheck
sys.path.append(os.path.join(sys.path[0],"../"))
import Psql

import checkerconfig

# Variables

log = ''
dbname = ''

checkercfg = ''

output_folder = "XXX/string" # I recommend that you name this folder string.

CS = 700

# Functions



def read_string():

    string_file = checkerconfig.string_network_file
    log.info(  "Reading %s..." % string_file)
    

    string = [j for j in [i.rstrip().split() for i in open(string_file).readlines()[1:]] if int(j[2]) >= CS and j[0] != j[1]]

    return string


def select_uniprotkb(protein):

    R = Psql.qstring( "SELECT source FROM uniprotkb_protein WHERE uniprot_ac = '" + protein + "'",dbname)
    
    if len(R):
        return R[0][0]
    else:
        return ''


def ENSP2Uniprot(string):

    ensp2Uniprot = collections.defaultdict(list)
    string_mapped = []
    ambiguous     = []
    notmapped     = []
    ints_amb      = []
    
    map_file = checkerconfig.string_tab_file
    log.info(  'Reading %s...' % map_file)
    for l in open(map_file):
        l = l.split('\t')
        if len(l[20]) > 0:
            for p in l[20].split(';'):
                ensp2Uniprot[p.replace(' ','')].append(l[0])
    for i in string:
        p1 = i[0].replace('9606.','')
        p2 = i[1].replace('9606.','')
        if len(ensp2Uniprot[p1]) == 1 and len(ensp2Uniprot[p2]) == 1:
            string_mapped.append((ensp2Uniprot[p1][0],ensp2Uniprot[p2][0],{'weigth':i[2]}))
        else:
            if len(ensp2Uniprot[p1]) > 1:
                ambiguous.append(p1)
                ints_amb.append((p1,p2,{'weigth':i[2]}))
            elif len(ensp2Uniprot[p1]) == 0:
                notmapped.append(p1)
            if len(ensp2Uniprot[p2]) > 1:
                ints_amb.append((p1,p2,{'weigth':i[2]}))
                ambiguous.append(p2)
            elif len(ensp2Uniprot[p2]) == 0:
                notmapped.append(p2)
    log.info(  '\tProteins not mapped: %s' % len(list(set(notmapped))))
    sprot = []
    for p in list(set(ambiguous)):
        for i in ensp2Uniprot[p]:
            if select_uniprotkb(i) == 'sprot':
                sprot.append(i)
    for i in ints_amb:
        if i[0] in ambiguous:
            uni1 = [p for p in ensp2Uniprot[i[0]] if p in sprot]
        else:
            uni1 = ensp2Uniprot[i[0]]
        if i[1] in ambiguous:
            uni2 = [p for p in ensp2Uniprot[i[1]] if p in sprot]
        else:
            uni2 = ensp2Uniprot[i[1]]

        if len(uni1) and len(uni2):
            for t in itertools.product(uni1,uni2):
                t += ({'weigth':i[2]},)
                string_mapped.append(t)
    return string_mapped


def write_network(string):

	with open(output_folder + "/interactions.tsv", "w") as f:
		for s in string:
			if s[0] == s[1]: continue
			s = sorted([s[0], s[1]])
			f.write("%s\t%s\n" % (s[0], s[1]))

# Main

def main():
    
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)
  
    configFilename = sys.argv[2]
    
    global log,checkercfg

    checkercfg = checkerconfig.checkerConf(configFilename )  

    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    downloadsdir = checkercfg.getDirectory( "downloads" )
    
    global dbname,output_folder
    
    networksdir = checkercfg.getDirectory( "networks" )
    
    dirname = "string"
    output_folder = os.path.join(networksdir,dirname)
    check_dir = os.path.exists(output_folder)


    if check_dir == False:
        c = os.makedirs(output_folder)
    
    dbname = checkercfg.getVariable('UniprotKB', 'dbname'     )
    
	log.info(  "Reading STRING")
    string = read_string()
    log.info(  "Mapping to UniProtKB")
    string = ENSP2Uniprot(string)
    log.info(  "Writing network")
    write_network(string)


if __name__ == '__main__':
	main()
