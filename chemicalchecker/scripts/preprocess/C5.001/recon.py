import xml.etree.ElementTree as pxml
import os
import collections
import itertools as itt
import networkx as nx
import numpy as np
import operator
from chemicalchecker.util import logged
from chemicalchecker.util import HotnetNetwork
from chemicalchecker.util import psql
from chemicalchecker.database import Dataset


@logged
class recon():
    """Validation generator class.

    Creates a validation set
    """

    def __init__(self, net_dir, dbname, cpu=1):
        """Initialize the validation.

        Args:
            net_dir(str): The path to directory where to save network data.
            dbname(str): Name of the DB that needs to be used
        """
        self.dbname = "uniprotkb_" + dbname
        self.net_dir = net_dir
        self.cpu = cpu

# Functions

    def _is_sprot(self, protein):

        R = psql.qstring(
            "SELECT source FROM uniprotkb_protein WHERE uniprot_ac = '" + protein + "'", self.dbname)

        if R[0][0] == 'sprot':
            return 1
        else:
            return 0

    def _metabolic_network_directed(self, filter, degree, chemicals, proteins, reactions, hgnc_mapping_file, genesid):

        geneid2Uniprot = self._uniprot_mapping(genesid, hgnc_mapping_file)
        self.__log.info('Generating metabolic network (directed)...')
        proteins_reaction = collections.defaultdict(dict)
        for r in reactions.values():
            for p in r['proteins']:
                if len(proteins_reaction[p]) == 0:
                    proteins_reaction[p]['reactants'] = set()
                    proteins_reaction[p]['products'] = set()

                proteins_reaction[p]['reactants'] |= set(r['reactants'])
                proteins_reaction[p]['products'] |= set(r['products'])
        edges_enz = []
        edges = []
        uniprot = collections.defaultdict(list)
        for pi in proteins:
            for g in set(proteins[pi]['genesid']):
                if len(geneid2Uniprot[g]) == 1:
                    uniprot[pi].extend(geneid2Uniprot[g])
                elif len(geneid2Uniprot[g]) > 1:
                    ok = []
                    [ok.append(u)
                     for u in geneid2Uniprot[g] if self._is_sprot(u)]
                    if len(ok) > 0:
                        uniprot[pi].extend(ok)
                    else:
                        uniprot[pi].extend(geneid2Uniprot[g])
            uniprot[pi] = list(set(uniprot[pi]))
            if len(set(proteins[pi]['genesid'])) > 1:
                for d in itt.combinations(uniprot[pi], 2):
                    edges_enz.append((d[0], d[1], {'weight': 1}))
        for pair in itt.combinations(proteins_reaction.keys(), 2):

            common = proteins_reaction[pair[0]][
                'reactants'] & proteins_reaction[pair[1]]['products']
            common |= proteins_reaction[pair[1]][
                'reactants'] & proteins_reaction[pair[0]]['products']
            common -= filter

            if len(common) > 0:
                for i in uniprot[pair[0]]:
                    for j in uniprot[pair[1]]:
                        edges.append(
                            (i, j, {'weight': 1. / sorted([len(chemicals[m]['proteins']) for m in common])[0]}))
        self.__log.info('DONE')
        G = self._generate_network(edges)
        self.__log.info('DONE')
        G.add_edges_from(edges_enz)
        self.__log.info('DONE')
        return G

    def _generate_network(self, ppis):

        G = nx.Graph()
        G.add_edges_from([p for p in ppis if p[0] != p[1]])
        self.__log.info('\tNumber of edges: %s' % G.number_of_edges())
        self.__log.info('\tNumber of nodes: %s' % G.number_of_nodes())
        self.__log.info('\tDensity: %s' % nx.density(G))
        self.__log.info('\tDegree:')
        self.__log.info('\t\tAverage: %s' % np.mean(
            [val for (node, val) in G.degree()]))
        self.__log.info('\t\tMedian: %s' % np.median(
            [val for (node, val) in G.degree()]))
        self.__log.info('\tAverage cluster: %s' % nx.average_clustering(G))
        self.__log.info('\tTransitivity: %s' % nx.transitivity(G))
        return G

    def _get_filter(self, chemicals, reactions):
        THRESHOLD = 0.01
        stats = collections.defaultdict(int)
        for c in chemicals:
            for v in reactions.values():
                if c in v['reactants'] or c in v['products']:
                    stats[c] += 1
        degree = {}
        for i in set([i for i in sorted(stats.items(), key=operator.itemgetter(1))]):
            degree[i[0]] = i[1]

        to_remove = int(len(set(degree.keys())) * THRESHOLD)

        return set([i[0] for i in sorted(degree.items(), key=operator.itemgetter(1), reverse=True)[to_remove:]]), degree

    def _read_xml(self, chemicals, proteins, reactions, genesid):

        xml_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "files", 'recon_2.2.xml')
        self.__log.info('Reading %s...' % xml_file)
        xml = pxml.parse(xml_file)
        root = xml.getroot()
        for model in root:
            for species in model.findall('{http://www.sbml.org/sbml/level2/version4}listOfSpecies'):
                for specie in species.findall('{http://www.sbml.org/sbml/level2/version4}species'):
                    if specie.attrib['sboTerm'] == 'SBO:0000247':  # Simple chemical
                        if 'name' in specie.attrib.keys():
                            chemicals[specie.attrib['id']][
                                'name'] = specie.attrib['name']
                        chemicals[specie.attrib['id']]['proteins'] = set()
                        for annotation in specie.iter():
                            if annotation.tag == '{http://www.w3.org/1999/xhtml}p':
                                if annotation.text.find('FORMULA') > -1:
                                    chemicals[specie.attrib['id']][
                                        'formula'] = annotation.text.replace('FORMULA: ', '')
            for reactions_ in model.findall('{http://www.sbml.org/sbml/level2/version4}listOfReactions'):
                for reaction in reactions_.findall('{http://www.sbml.org/sbml/level2/version4}reaction'):
                    reactions[reaction.attrib['id']][
                        'name'] = reaction.attrib['name']
                    reactions[reaction.attrib['id']]['reactants'] = []
                    reactions[reaction.attrib['id']]['products'] = []
                    reactions[reaction.attrib['id']]['proteins'] = []
                    for annotation in reaction.iter():
                        if annotation.tag == '{http://www.w3.org/1999/xhtml}p' and annotation.text.find('GENE_ASSOCIATION:') > -1:
                            if len(annotation.text.split('GENE_ASSOCIATION:')[1]) > 1:
                                reactions[reaction.attrib['id']][
                                    'rules'] = annotation.text.split('GENE_ASSOCIATION:')[1]
                                r = reactions[reaction.attrib['id']]['rules'].replace(
                                    ' ', '').replace('(', '').replace(')', '')
                                prots = r.split('or')
                                reactions[reaction.attrib['id']][
                                    'proteins'] = prots
                                for p in prots:
                                    if p not in proteins.keys():
                                        proteins[p]['uniprot'] = []
                                        proteins[p]['genesid'] = p.split('and')
                                    genesid.extend(proteins[p]['genesid'])
                        elif annotation.tag == '{http://www.sbml.org/sbml/level2/version4}listOfReactants':
                            [reactions[reaction.attrib['id']]['reactants'].append(reactant.attrib['species']) for reactant in annotation.findall(
                                '{http://www.sbml.org/sbml/level2/version4}speciesReference')]
                        elif annotation.tag == '{http://www.sbml.org/sbml/level2/version4}listOfProducts':
                            [reactions[reaction.attrib['id']]['products'].append(product.attrib['species']) for product in annotation.findall(
                                '{http://www.sbml.org/sbml/level2/version4}speciesReference')]

                    for m in reactions[reaction.attrib['id']]['reactants']:
                        chemicals[m]['proteins'] |= set(
                            reactions[reaction.attrib['id']]['proteins'])
                    for m in reactions[reaction.attrib['id']]['products']:
                        chemicals[m]['proteins'] |= set(
                            reactions[reaction.attrib['id']]['proteins'])
        return self._get_filter(chemicals, reactions)

    def _uniprot_mapping(self, genesid, hgnc_mapping_file):

        geneid2Uniprot = collections.defaultdict(list)
        self.__log.info('Mapping GeneID to UniprotAC...')
        genes = list(set(genesid))
        self.__log.info('\t# Unique GeneID: %s' % len(genes))
        self.__log.info('\tRetrieving reference proteome...')
        proteome = self._select_reference_proteome()
        self.__log.info('\tReading %s...' % hgnc_mapping_file)
        for l in open(hgnc_mapping_file).readlines()[1:]:
            l = l.split('\t')
            if len(l[25]) > 0:
                for p in l[25].split('|'):
                    if p.replace(' ', '') in proteome:
                        geneid2Uniprot[l[0]].append(p.replace(' ', ''))
        return geneid2Uniprot

    def _select_reference_proteome(self):

        R = psql.qstring(
            "SELECT uniprot_ac FROM uniprotkb_protein WHERE taxid = '9606' AND complete = 'Complete proteome' AND reference = 'Reference proteome'", self.dbname)

        proteome = [i[0] for i in R]
        return proteome

    def _write_network(self, G):

        E = set()
        for e in G.edges():
            E.update([tuple(sorted([e[0], e[1]]))])
        with open(self.net_dir + "/interactions.tsv", "w") as f:
            for e in E:
                f.write("%s\t%s\n" % (e[0], e[1]))

    def run(self):
        """Run the network script."""

        chemicals = collections.defaultdict(dict)
        proteins = collections.defaultdict(dict)
        reactions = collections.defaultdict(dict)
        genesid = []

        dataset_code = 'C5.001'
        dataset = Dataset.get(dataset_code)
        map_files = {}
        for ds in dataset.datasources:
            map_files[ds.name] = ds.data_path

        hgnc_mapping_file = os.path.join(
            map_files["hgnc_complete"], "hgnc_complete_set.txt")

        filterd, degree = self._read_xml(
            chemicals, proteins, reactions, genesid)
        G = self._metabolic_network_directed(
            filterd, degree, chemicals, proteins, reactions, hgnc_mapping_file, genesid)
        self._write_network(G)

        readyfile = "recon.ready"

        HotnetNetwork.prepare(self.net_dir + "/interactions.tsv", self.net_dir, self.cpu)

        with open(os.path.join(self.net_dir, readyfile), "w") as f:
            f.write("")
