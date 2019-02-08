# Read a network file and prepare files.
# Compute influence matrices, etc.

# Imports

import os
import sys, argparse
from tqdm import tqdm
import networkx as nx
import subprocess

# Variables


# Parse arguments

def get_parser():
    description = 'Parse interactions, prepare files and compute beta and influence matrices.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--interactions', type=str, required=True, help='Interactions filename')
    parser.add_argument('-a', '--all_nodes', type=bool, default=False, required=False, help='By default, we only get the giant component')
    parser.add_argument('-o', '--output_folder', type=str, required=False, default='.', help='Output folder')
    parser.add_argument('-p', '--hotnet_path', type=str, required=False, default='.', help='Hotnet path')
    return parser

args = get_parser().parse_args(sys.argv[1:])

# Read network

print "Reading network"

G = nx.Graph()

with open(args.interactions, "r") as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        G.add_edge(l[0], l[1])

if not args.all_nodes:
    G = max(nx.connected_component_subgraphs(G), key=len)

# Writing files

# Index-to-gene file

f = open("%s/idx2node.tsv" % args.output_folder, "w")
i = 1
node_idx = {}
for n in G.nodes():
    f.write("%d\t%s\n" % (i, n))
    node_idx[n] = i
    i += 1
f.close()

# Edge-list file

f = open("%s/edgelist.tsv" % args.output_folder, "w")
for e in G.edges():
    f.write("%d\t%d\n" % (node_idx[e[0]], node_idx[e[1]]))
f.close()


# Calculate beta

print "Computing beta"

cmd = "python %s/choose_beta.py -i %s -o %s" % (args.hotnet_path, args.output_folder+"/edgelist.tsv", args.output_folder+"/beta.txt")
subprocess.Popen(cmd, shell = True).wait()


# Calculate similarity matrix

print "Similarity matrix"
b = float(open(args.output_folder+"/beta.txt", "r").read())
cmd = "python %s/create_similarity_matrix.py -i %s -b %.2f -o %s" % (args.hotnet_path, args.output_folder+"/edgelist.tsv", b, args.output_folder+"/similarity_matrix.h5")
subprocess.Popen(cmd, shell = True).wait()
