from gensim.models import Word2Vec
from chemicalchecker.util import logged
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import h5py
import numpy as np


@logged
class Word2VecWrapper():

    def __init__(self, recipe):
        self.__log.info("INIT")
        self.recipe = recipe

    def learn_embedding(self, corpus):
        embedding = Word2Vec(
            corpus,
            size=self.recipe.params.embedding.dimensions,
            window=self.recipe.params.embedding.window_size,
            min_count=self.recipe.params.embedding.min_count,
            workers=self.recipe.params.embedding.workers,
            sg=1)
        self.word_vectors = embedding.wv

    def save(self, filepath):
        self.word_vectors.save(filepath)

    @classmethod
    def load(cls, filepath, embedding_format='gensim'):
        w2v = cls(None)
        if embedding_format == 'gensim':
            w2v.word_vectors = KeyedVectors.load(filepath, mmap='r')
        elif embedding_format == 'c_txt':
            w2v.word_vectors = KeyedVectors.load_word2vec_format(
                datapath(filepath), binary=False)
        elif embedding_format == 'c_bin':
            w2v.word_vectors = KeyedVectors.load_word2vec_format(
                datapath(filepath), binary=True)
        return w2v

    @staticmethod
    def convert(in_file, out_file, in_format='c_txt', out_format='h5', names_map=None, limit_words=set()):
        Word2VecWrapper.__log.info("Converting %s to %s" % (in_file, out_file))
        if in_format == 'c_txt':
            with open(in_file, 'r') as fh:
                words = list()
                vectors = list()
                fh.readline()  # skip first row
                skipped = 0
                for line in fh:
                    fields = line.split()
                    # first colum is id
                    word = int(fields[0])
                    if word in limit_words:
                        skipped += 1
                        continue
                    # then embedding
                    vector = np.fromiter((float(x) for x in fields[1:]),
                                         dtype=np.float)
                    words.append(word)
                    vectors.append(vector)
            # to numpy arrays
            words = np.array(words)
            matrix = np.array(vectors)
            # get them sorted
            sorted_idx = np.argsort(words)
            Word2VecWrapper.__log.info('words: %s' % str(words.shape))
            Word2VecWrapper.__log.info('matrix: %s' % str(matrix.shape))
            Word2VecWrapper.__log.info('skipped: %s' % skipped)
        else:
            raise Exception("Unrecognized input format.")

        if out_format == 'h5':
            names = np.loadtxt(names_map, dtype='|S27', usecols=[1])
            with h5py.File(out_file, "w") as fh:
                fh.create_dataset('inchikeys', data=names[words[sorted_idx]])
                fh.create_dataset('V', data=matrix[sorted_idx])

        else:
            raise Exception("Unrecognized output format.")
