"""Node2vec wrapper.

Use `Node2Vec <https://snap.stanford.edu/node2vec/>`_ to generate molecule
embeddings.
"""
from .word2vec import Word2VecWrapper as Word2Vec
from .node2vec import Node2Vec