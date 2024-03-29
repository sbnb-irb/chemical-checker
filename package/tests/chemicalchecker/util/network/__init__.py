"""Network representations.

Each class provides mean of loading and accessing a network/graph.

  * **NetworkX**: :mod:`~chemicalchecker.util.network.network.NetworkxNetwork`
    Wrapper around the pure Python
    `NetworkX package <https://networkx.github.io/>`_. This is very easy to use
    but becomes too expensive in memory.

  * **SNAPNetwork**: :mod:`~chemicalchecker.util.network.network.SNAPNetwork`
    Wrapper around the C++/Python-binded
    `SNAP library <https://github.com/snap-stanford/snap>`_. Scales to large
    network with millions of nodes and billions od edges.

  * **HotnetNetwork**:
    :mod:`~chemicalchecker.util.network.network.HotnetNetwork`
    Network implementation that is suited for
    `HotNet2 module <https://github.com/raphael-group/hotnet2>`_.

  * **MultiEdgeNetwork**:
    :mod:`~chemicalchecker.util.network.network.MultiEdgeNetwork`
    experimental, multi edge and multi mode in SNAP.
"""
from .network import NetworkxNetwork
from .network import MultiEdgeNetwork
from .network import SNAPNetwork
from .network import HotnetNetwork
