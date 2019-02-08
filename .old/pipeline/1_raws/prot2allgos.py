import sys, os
import networkx as nx
import collections

uniprot_ac = sys.argv[1]
myfold = sys.argv[2]
goa_file = sys.argv[3]


# Read Graph
G = nx.DiGraph()
f = open(myfold+"/bp.tsv", "r")
for l in f:
    l = l.rstrip("\n").split("\t")
    G.add_edge(l[0], l[1])
f.close()

# Read Uniprot GOA
uniprot_go = collections.defaultdict(set)
f = open(goa_file, "r")
for l in f:
    if l[0] == "!": continue
    l = l.split("\t")
    if l[8] != "P": continue
    uniprot = l[1]
    go = l[4]
    uniprot_go[uniprot].update([go])
f.close()

bp = "GO:0008150"

if uniprot_ac not in uniprot_go: sys.exit("%s not found in GO-BPs" % uniprot_ac)
P = set()
for go in uniprot_go[uniprot_ac]:
    path = set()
    p  = [y for x in nx.all_simple_paths(G, bp, go) for y in x]
    P.update(p)

f = open(myfold+"/prot_allgos/%s.tsv" % uniprot_ac, "w")
for p in P:
    f.write("%s\n" % p)
f.close()
