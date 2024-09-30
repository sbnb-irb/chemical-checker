"""Network representations.

Each class provides mean of accessing a network/graph.
"""
import os
import json
import numpy as np
import networkx as nx

from chemicalchecker.util import logged, Config


@logged
class NetworkxNetwork():
    """NetworkxNetwork class.

    Network representation with NetworkX.
    Simple but heavy in memory.
    Single type of node, single type of edge.
    """

    def __init__(self, network):
        """Initialize a NetworkxNetwork instance."""
        self._network = network
        self.__log.info("Nodes: %s Edges: %s" % (
            len(self._network.nodes()),
            len(self._network.edges())))

    @property
    def nodes(self, data=False):
        return self._network.nodes(data=data)

    def get_node(self, node):
        return self._network[node]

    def edges(self, data=False):
        return self._network.edges(data=data)

    def neighbors(self, node):
        neighbors = list()
        for node, edge_data in self._network[node].items():
            node_data = self._network.nodes[node]
            neighbors.append((edge_data, node, node_data))
        return neighbors

    def has_edge(self, node, previous):
        return self._network.has_edge(node, previous)


@logged
class MultiEdgeNetwork():
    """MultiEdgeNetwork class.

    Multimodal network representation with SNAP.
    Multiple type of node, Multiple type of edge.
    """

    def __init__(self, network):
        """Initialize a MultiEdgeNetwork instance."""
        try:
            import sys
            snapPath = Config().TOOLS.asdict()['snap']
            sys.path.append( snapPath )
            
            import snap
            self.snap = snap
        except ImportError:
            raise ImportError("requires snap " +
                              "http://snap.stanford.edu/")
        self._network = network
        self.__log.info("Node types: %s" % network.GetModeNets())
        self.node_types = list()
        modeneti = network.BegModeNetI()
        while modeneti < network.EndModeNetI():
            self.__log.info("Nodes in '{}': {:>12}".format(
                network.GetModeName(modeneti.GetModeId()),
                modeneti.GetModeNet().GetNodes()))
            modenet_name = network.GetModeName(modeneti.GetModeId())
            self.node_types.append(modenet_name)
            self._current_modenet = (
                modenet_name, network.GetModeNetByName(modenet_name))
            modeneti.Next()
        self.__log.info("Link types: %s" % network.GetCrossNets())
        self.edge_types = list()
        crossneti = network.BegCrossNetI()
        while crossneti < network.EndCrossNetI():
            self.__log.info("Edges in '{}': {:>12}".format(
                network.GetCrossName(crossneti.GetCrossId()),
                crossneti.GetCrossNet().GetEdges()))
            crossnet_name = network.GetCrossName(crossneti.GetCrossId())
            self.edge_types.append(crossnet_name)
            self._current_crossnet = (
                crossnet_name, network.GetCrossNetByName(crossnet_name))
            crossneti.Next()
        self._neighbors = dict()

    def nodes(self, node_type=None, data=False):
        if not node_type:
            node_type = self.node_types[0]
        modenet = self._network.GetModeNetByName(node_type)
        node = modenet.BegMMNI()
        while node < modenet.EndMMNI():
            if data:
                attr = self.snap.TStrV()
                node.GetStrAttrVal(attr)
                yield (node.GetId(), attr[0])
            else:
                yield node.GetId()
            node.Next()

    def edges(self, edge_type=None, data=False):
        if not edge_type:
            edge_type = self.edge_types[0]
        crossnet = self._network.GetCrossNetByName(edge_type)
        edge = crossnet.BegEI()
        while edge < crossnet.EndEI():
            src = edge.GetSrcNId()
            dst = edge.GetDstNId()
            if data:
                weight = self.snap.TFltV()
                crossnet.FltAttrValueEI(edge.GetId(), weight)
                yield (src, dst, weight[0])
            else:
                yield (src, dst)
            edge.Next()

    def neighbors(self, nodeid, edge_type, node_type, data=True):
        if self._current_modenet[0] == node_type:
            modenet = self._current_modenet[1]
        else:
            modenet = self._network.GetModeNetByName(node_type)
        if self._current_crossnet[0] == edge_type:
            crossnet = self._current_crossnet[1]
        else:
            crossnet = self._network.GetCrossNetByName(edge_type)
        edgeids = self.snap.TIntV()
        modenet.GetNeighborsByCrossNet(nodeid, edge_type, edgeids, True)
        for edgeid in edgeids:
            ei = crossnet.GetEdgeI(edgeid)
            edge = self.snap.TCrossNetEdgeI(ei)
            ei.disown()
            if data:
                weight = self.snap.TFltV()
                crossnet.FltAttrValueEI(edgeid, weight)
                yield (edge.GetDstNId(), weight[0])
            else:
                yield edge.GetDstNId()

    def has_edge(self, src, dst, edge_type, node_type):
        neighbors = self.neighbors(src, edge_type, node_type, data=False)
        for nodeid in neighbors:
            if dst == nodeid:
                return True
        return False

    def out_degree(self, nodeid, edge_type, node_type):
        neighbors = self.neighbors(nodeid, edge_type, node_type)
        return len(neighbors)

    def print_nodes(self, node_type):
        for node in self.nodes(node_type, data=True):
            print(node)

    def print_edges(self, edge_type):
        for edge in self.edges(edge_type, data=True):
            print(edge)


@logged
class SNAPNetwork():
    """SNAPNetwork class.

    Network representation with SNAP.
    Single type of node, single type of edge.
    """

    def __init__(self, network):
        """Initialize a SNAPNetwork instance."""
        try:
            import sys
            sys.path.append( Config().TOOLS.snap )
            import snap
            self.snap = snap
        except ImportError:
            raise ImportError("requires snap " +
                              "http://snap.stanford.edu/")
        self._network = network
        self.__log.info("Nodes : {:>12}".format(network.GetNodes()))
        self.__log.info("Edges : {:>12}".format(network.GetEdges()))

    @classmethod
    def from_file(cls, filename, delimiter=' ', read_weights=True):
        try:
            import sys
            sys.path.append( Config().TOOLS.snap )
            import snap
        except ImportError:
            raise ImportError("requires snap " +
                              "http://snap.stanford.edu/")
        filename = os.path.abspath(filename)
        network = snap.LoadEdgeList(snap.TNEANet, filename, 0, 1, delimiter)
        # add weigths
        if read_weights:
            with open(filename, 'r') as fh:
                eid = 0
                for line in fh:
                    network.AddFltAttrDatE(
                        eid, float(line.split()[2]), 'weight')
                    eid += 1
        snapnet = cls(network)
        return snapnet

    def nodes(self, data=False):
        node = self._network.BegNI()
        while node < self._network.EndNI():
            if data:
                attr = self.snap.TStrV()
                node.GetStrAttrVal(attr)
                yield (node.GetId(), attr[0])
            else:
                yield node.GetId()
            node.Next()

    def edges(self, data=False):
        edge = self._network.BegEI()
        while edge < self._network.EndEI():
            src = edge.GetSrcNId()
            dst = edge.GetDstNId()
            if data:
                weight = self.snap.TFltV()
                self._network.FltAttrValueEI(edge.GetId(), weight)
                yield (src, dst, weight[0])
            else:
                yield (src, dst)
            edge.Next()

    def neighbors(self, nodeid, data=False):
        node = self._network.GetNI(nodeid)
        for nid in range(node.GetOutDeg()):
            neig = node.GetNbrNId(nid)
            if data:
                edgeid = node.GetNbrEId(nid)
                weight = self.snap.TFltV()
                self._network.FltAttrValueEI(edgeid, weight)
                yield (neig, weight[0])
            else:
                yield neig

    def has_edge(self, src, dst):
        neighbors = self.neighbors(src)
        for nodeid in neighbors:
            if dst == nodeid:
                return True
        return False

    def out_degree(self, nodeid):
        node = self._network.GetNI(nodeid)
        return node.GetOutDeg()

    def print_nodes(self):
        for node in self.nodes():
            print(node)

    def print_edges(self):
        for edge in self.edges():
            print(edge)

    def save(self, filename):
        FOut = self.snap.TFOut(filename)
        self._network.Save(FOut)
        FOut.Flush()

    def stats_toJSON(self, filename):
        stats = dict()
        stats["nodes"] = self._network.GetNodes()
        stats["edges"] = self._network.GetEdges()
        zeroNodes = 0
        zeroInNodes = 0
        zeroOutNodes = 0
        nonZIODegNodes = 0
        node = self._network.BegNI()
        degrees = list()
        while node < self._network.EndNI():
            degrees.append(node.GetDeg())
            if (node.GetDeg() == 0):
                zeroNodes += 1
            if (node.GetInDeg() == 0):
                zeroInNodes += 1
            if (node.GetOutDeg() == 0):
                zeroOutNodes += 1
            if (node.GetInDeg() != 0 & node.GetOutDeg() != 0):
                nonZIODegNodes += 1
            node.Next()
        edge = self._network.BegEI()
        weights = list()
        while edge < self._network.EndEI():
            weight = self.snap.TFltV()
            self._network.FltAttrValueEI(edge.GetId(), weight)
            weights.append(weight[0])
            edge.Next()
        # nodes without edges?
        stats["zeroNodes"] = zeroNodes
        stats["zeroInNodes"] = zeroInNodes
        stats["zeroOutNodes"] = zeroOutNodes
        stats["nonZIODegNodes"] = nonZIODegNodes
        # degree distribution
        stats["Degree_min"] = min(degrees)
        stats["Degree_max"] = max(degrees)
        stats["Degree_25"] = np.percentile(degrees, 25)
        stats["Degree_50"] = np.percentile(degrees, 50)
        stats["Degree_75"] = np.percentile(degrees, 75)
        # weights distribution
        stats["Weight_min"] = min(weights)
        stats["Weight_max"] = max(weights)
        stats["Weight_25"] = np.percentile(weights, 25)
        stats["Weight_50"] = np.percentile(weights, 50)
        stats["Weight_75"] = np.percentile(weights, 75)
        # fraction of nodes in largest weakly connected component
        stats["WccSz"] = self.snap.GetMxWccSz(self._network)
        # fraction of nodes in largest strongly connected component
        stats["SccSz"] = self.snap.GetMxSccSz(self._network)

        with open(filename, 'w') as fh:
            json.dump(stats, fh)


@logged
class HotnetNetwork():
    """HotnetNetwork class.

    Network tools for hotnet.
    Read network and create files.
    """

    def __init__(self, network):
        """Initialize a HotnetNetwork instance."""
        self._network = network
        self.__log.info("Nodes: %s Edges: %s" % (
            len(self._network.nodes()),
            len(self._network.edges())))

    @staticmethod
    def prepare(interactions, out_path, hotnet, all_nodes=False):

        HotnetNetwork.__log.info("Reading network")

        G = nx.Graph()

        with open(interactions, "r") as f:
            for l in f:
                l = l.rstrip("\n").split("\t")
                G.add_edge(l[0], l[1])

        if not all_nodes:

            G = max(list(G.subgraph(c)
                         for c in nx.connected_components(G)), key=len)

        # Writing files

        # Index-to-gene file

        f = open("%s/idx2node.tsv" % out_path, "w")
        i = 1
        node_idx = {}
        for n in G.nodes():
            f.write("%d\t%s\n" % (i, n))
            node_idx[n] = i
            i += 1
        f.close()

        # Edge-list file

        f = open("%s/edgelist.tsv" % out_path, "w")
        for e in G.edges():
            f.write("%d\t%d\n" % (node_idx[e[0]], node_idx[e[1]]))
        f.close()

        # Calculate beta
        print('Compute beta - ', 'edges:', len( list(G.edges()) ), 'nodes:', len( list(G.nodes()) ) )

        HotnetNetwork.__log.info("Computing beta")

        hotnet.choose_beta(os.path.join(out_path, "edgelist.tsv"),
                           os.path.join(out_path, "beta.txt"))

        # Calculate similarity matrix

        HotnetNetwork.__log.info("Calculate Similarity matrix")
        b = float(open(os.path.join(out_path, "beta.txt"), "r").read())
        hotnet.create_similarity_matrix(os.path.join(out_path, "edgelist.tsv"),
                                        os.path.join(out_path, "similarity_matrix.h5"), b=b)
