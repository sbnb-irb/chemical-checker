#!/miniconda/bin/python


'''

Fetch Recon data.

Based, mainly, in Tere's scripts.

'''

# Imports

import xml.etree.ElementTree as pxml
import os
import collections
import itertools as itt
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

#### ALL OF THIS COMES FROM TERE! 

OUTPUTDIR   = 'recon2_v2'
THRESHOLD   = 0.01 # Percentage of metabolites to be removed
log = ''
dbname = ''

chemicals = collections.defaultdict(dict)
proteins  = collections.defaultdict(dict)
reactions = collections.defaultdict(dict)
genesid   = []

#### 



# Functions

def is_sprot(protein):

    R = Psql.qstring( "SELECT source FROM uniprotkb_protein WHERE uniprot_ac = '" + protein + "'",dbname)
    
    if R[0][0] == 'sprot':
        return 1
    else:
        return 0


def metabolic_network_directed(filter,degree):
    
    global chemicals,proteins,reactions,genesid

    degreeUniprot = {}
    geneid2Uniprot = uniprot_mapping()
    #log.info(  geneid2Uniprot.items()[:100])
    log.info(  'Generating metabolic network (directed)...')
    proteins_reaction = collections.defaultdict(dict)
    for r in reactions.values():
        for p in r['proteins']:
            if len(proteins_reaction[p]) == 0:
                proteins_reaction[p]['reactants'] = set()
                proteins_reaction[p]['products'] = set()

            proteins_reaction[p]['reactants'] |= set(r['reactants'])
            proteins_reaction[p]['products'] |= set(r['products'])
    edges_enz = []
    edges     = []
    uniprot= collections.defaultdict(list)
    for pi in proteins:
        for g in set(proteins[pi]['genesid']):
            if len(geneid2Uniprot[g]) == 1:
                uniprot[pi].extend(geneid2Uniprot[g])
            elif len(geneid2Uniprot[g]) > 1:
                ok = []
                [ok.append(u) for u in geneid2Uniprot[g] if is_sprot(u)]
                if len(ok) > 0:
                    uniprot[pi].extend(ok)
                else:
                    uniprot[pi].extend(geneid2Uniprot[g])
        uniprot[pi] = list(set(uniprot[pi]))
        if len(set(proteins[pi]['genesid'])) > 1:
            for d in itt.combinations(uniprot[pi],2):
                edges_enz.append((d[0],d[1], {'weight':1}))
    for pair in itt.combinations(proteins_reaction.keys(),2):

        common  = proteins_reaction[pair[0]]['reactants'] & proteins_reaction[pair[1]]['products']
        common |= proteins_reaction[pair[1]]['reactants'] & proteins_reaction[pair[0]]['products']
        common -= filter

        if len(common) > 0:
            for i in uniprot[pair[0]]:
                for j in uniprot[pair[1]]:
                    edges.append((i,j, {'weight':1./sorted([len(chemicals[m]['proteins']) for m in common])[0]}))
    log.info(  'DONE')
    G = config.generate_network(edges)
    log.info(  'DONE')
    G.add_edges_from(edges_enz)
    log.info(  'DONE')
    return G





def get_filter():
    global chemicals,proteins,reactions,genesid
    stats = collections.defaultdict(int)
    for c in chemicals:
        for v in reactions.values():
            if c in v['reactants'] or c in v['products']:
                stats[c] += 1
    degree = {}
    for i in set([i for i in sorted(stats.items(), key=operator.itemgetter(1))]):
        degree[i[0]] = i[1]

    chemicals = set(degree.keys())

    to_remove = int(len(chemicals)*THRESHOLD)

    return set([i[0] for i in sorted(degree.items(), key=operator.itemgetter(1), reverse=True)[to_remove:]]), degree


def read_xml():

    global chemicals,proteins,reactions,genesid
    xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../files/",'recon_2.2.xml')
    log.info(  'Reading %s...' % xml_file)
    xml = pxml.parse(xml_file)
    root = xml.getroot()
    for model in root:
        for species in model.findall('{http://www.sbml.org/sbml/level2/version4}listOfSpecies'):
            for specie in species.findall('{http://www.sbml.org/sbml/level2/version4}species'):
                if specie.attrib['sboTerm'] == 'SBO:0000247': #Simple chemical
                    if 'name' in specie.attrib.keys(): chemicals[specie.attrib['id']]['name'] = specie.attrib['name']
                    chemicals[specie.attrib['id']]['proteins'] = set()
                    for annotation in specie.iter():
                        if annotation.tag =='{http://www.w3.org/1999/xhtml}p':
                            if annotation.text.find('FORMULA') > -1:
                                chemicals[specie.attrib['id']]['formula'] = annotation.text.replace('FORMULA: ','')
        for reactions_ in model.findall('{http://www.sbml.org/sbml/level2/version4}listOfReactions'):
            for reaction in reactions_.findall('{http://www.sbml.org/sbml/level2/version4}reaction'):
                reactions[reaction.attrib['id']]['name']      = reaction.attrib['name']
                reactions[reaction.attrib['id']]['reactants'] = []
                reactions[reaction.attrib['id']]['products']  = []
                reactions[reaction.attrib['id']]['proteins']  = []
                for annotation in reaction.iter():
                    if annotation.tag == '{http://www.w3.org/1999/xhtml}p' and annotation.text.find('GENE_ASSOCIATION:') >  -1:
                        if len(annotation.text.split('GENE_ASSOCIATION:')[1]) > 1:
                            reactions[reaction.attrib['id']]['rules'] = annotation.text.split('GENE_ASSOCIATION:')[1]
                            r = reactions[reaction.attrib['id']]['rules'].replace(' ','').replace('(','').replace(')','')
                            prots = r.split('or')
                            reactions[reaction.attrib['id']]['proteins'] = prots
                            for p in prots:
                                if p not in proteins.keys():
                                    proteins[p]['uniprot'] = []
                                    proteins[p]['genesid'] = p.split('and')
                                genesid.extend(proteins[p]['genesid'])
                    elif annotation.tag == '{http://www.sbml.org/sbml/level2/version4}listOfReactants':
                        [reactions[reaction.attrib['id']]['reactants'].append(reactant.attrib['species']) for reactant in annotation.findall('{http://www.sbml.org/sbml/level2/version4}speciesReference')]
                    elif annotation.tag == '{http://www.sbml.org/sbml/level2/version4}listOfProducts':
                        [reactions[reaction.attrib['id']]['products'].append(product.attrib['species']) for product in annotation.findall('{http://www.sbml.org/sbml/level2/version4}speciesReference')]

                for m in reactions[reaction.attrib['id']]['reactants']:
                    chemicals[m]['proteins'] |= set(reactions[reaction.attrib['id']]['proteins'])
                for m in reactions[reaction.attrib['id']]['products']:
                    chemicals[m]['proteins'] |= set(reactions[reaction.attrib['id']]['proteins'])
    return get_filter()

def uniprot_mapping():

    geneid2Uniprot = collections.defaultdict(list)
    log.info(  'Mapping GeneID to UniprotAC...')
    genes= list(set(genesid))
    log.info(  '\t# Unique GeneID: %s' % len(genes))
    log.info(  '\tRetrieving reference proteome...')
    #proteome = select_reference_proteome()
    R = Psql.qstring("select genename,uniprot_ac From uniprotkb_protein where taxid = '9606' and genename != '' and complete = 'Complete proteome' AND reference = 'Reference proteome'", dbname)


    for l in R: 
        geneid2Uniprot[l[0]].append(l[1])
   
    return geneid2Uniprot

def select_reference_proteome():

    R = Psql.qstring("SELECT uniprot_ac FROM uniprotkb_protein WHERE taxid = '9606' AND complete = 'Complete proteome' AND reference = 'Reference proteome'",dbname)
   
    proteome = [i[0] for i in R]
    return proteome


def write_network(G):
    E = set()
    for e in G.edges():
        E.update([tuple(sorted([e[0], e[1]]))])
    with open(OUTPUTDIR + "/interactions.tsv") as f:
        for e in E:
            f.write("%s\t%s\n" % (e[0], e[1]))


# Main

def main():
    
    if len(sys.argv) != 3:
        usage(sys.argv[0])
        sys.exit(1)
  
    configFilename = sys.argv[2]
    
    global log

    checkercfg = checkerconfig.checkerConf(configFilename )  

    logsFiledir = checkercfg.getDirectory( "logs" )

    log = logSystem(sys.stdout)

    downloadsdir = checkercfg.getDirectory( "downloads" )
    
    global dbname,OUTPUTDIR
    
    DATA =  os.path.join(downloadsdir,checkerconfig.pathway_data).replace('.gz','')
    networksdir = checkercfg.getDirectory( "networks" )
    
    dirname = "recon"
    OUTPUTDIR = os.path.join(networksdir,dirname)
    check_dir = os.path.exists(OUTPUTDIR)


    if check_dir == False:
        c = os.makedirs(OUTPUTDIR)
    
    dbname = checkercfg.getVariable('UniprotKB', 'dbname'     )

    filter, degree = read_xml()
    G = metabolic_network_directed(filter, degree)
    write_network(G)


if __name__ == '__main__':
    main()
