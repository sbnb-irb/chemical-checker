"""Wrapper for SNAP C++ implementation of node2vec.

An algorithmic framework for representational learning on graphs. [Aug 27 2018]
================================================================================
usage: node2vec
   -i:Input graph path (default:'graph/karate.edgelist')
   -o:Output graph path (default:'emb/karate.emb')
   -d:Number of dimensions. Default is 128 (default:128)
   -l:Length of walk per source. Default is 80 (default:80)
   -r:Number of walks per source. Default is 10 (default:10)
   -k:Context size for optimization. Default is 10 (default:10)
   -e:Number of epochs in SGD. Default is 1 (default:1)
   -p:Return hyperparameter. Default is 1 (default:1)
   -q:Inout hyperparameter. Default is 1 (default:1)
   -v Verbose output.
   -dr Graph is directed.
   -w Graph is weighted.
   -ow Output random walks instead of embeddings.


"""
import os
import distutils.spawn
import subprocess
from chemicalchecker.util import logged


@logged
class Node2Vec():
    """Wrapper to run SNAP's node2vec."""

    def __init__(self, executable="node2vec"):
        """Check if executable is found."""
        exec_file = distutils.spawn.find_executable(executable)
        if not exec_file:
            raise Exception("node2vec executable not found.")
        self.executable = exec_file

    def run(self, i, o, **kwargs):
        """Call external exe with given parameters.

        Args:
            i:Input graph path (default:'graph/karate.edgelist')
            o:Output graph path (default:'emb/karate.emb')
            d:Number of dimensions. Default is 128 (default:128)
            l:Length of walk per source. Default is 80 (default:80)
            r:Number of walks per source. Default is 10 (default:10)
            k:Context size for optimization. Default is 10 (default:10)
            e:Number of epochs in SGD. Default is 1 (default:1)
            p:Return hyperparameter. Default is 1 (default:1)
            q:Inout hyperparameter. Default is 1 (default:1)
            v Verbose output.
            dr Graph is directed.
            w Graph is weighted.
            ow Output random walks instead of embeddings.
        """
        # check input
        if not os.path.isfile(i):
            raise Exception("Input file not found.")

        # get arguments or default values
        d = kwargs.get("d", 128)
        l = kwargs.get("l", 80)
        r = kwargs.get("r", 10)
        k = kwargs.get("k", 10)
        e = kwargs.get("e", 1)
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)
        v = kwargs.get("v", True)
        ow = kwargs.get("ow", False)
        dr = kwargs.get("dr", True)
        w = kwargs.get("w", True)
        cpu = kwargs.get("cpu", 1)

        # prepare arguments
        args = [
            "-i:%s" % i,
            "-o:%s" % o,
            "-d:%s" % d,
            "-l:%s" % l,
            "-r:%s" % r,
            "-k:%s" % k,
            "-e:%s" % e,
            "-p:%s" % p,
            "-q:%s" % q,
        ]
        if v:
            args.append("-v")
        if dr:
            args.append("-dr")
        if w:
            args.append("-w")
        if ow:
            args.append("-ow")

        # this enables using as many CPU as required
        os.environ['OMP_NUM_THREADS'] = str(cpu)

        # log command
        self.__log.info(' '.join(args))
        self.__log.info("cpu: %s" % cpu)

        # run process
        process = subprocess.Popen([self.executable] + args,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # stream output as get generated
        for line in iter(process.stdout.readline, ''):
            self.__log.info(line.strip())

        for line in iter(process.stderr.readline, ''):
            self.__log.error(line.strip())
