from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import tensorflow as tf
import deepchem as dc
import h5py
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Label
from deepchem.models.tensorgraph.layers import Dense, SoftMax
from deepchem.models.tensorgraph.layers import L2Loss, WeightedError
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.feat.mol_graphs import ConvMol
from deepchem.data.data_loader import *
from deepchem.data.datasets import *
from deepchem.feat.graph_features import ConvMolFeaturizer
import pandas as pd
from chemicalchecker.util.splitter import Traintest
from chemicalchecker.core.signature_data import DataSignature
import json


class GraphSmilesSign():

    def __init__(self, models_path, **kwargs):

        self.models_path = os.path.abspath(models_path)
        if not os.path.exists(models_path):
            print("Specified models directory does not exist: %s" %
                  self.models_path)
            os.mkdir(self.models_path)

        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.batch_size = int(kwargs.get("batch_size", 1000))
        self.loss = kwargs.get("loss", 'mean_squared_error')
        self.epochs = int(kwargs.get("epochs", 200))
        self.cpu = kwargs.get("cpu", 32)

        self.path_eval = os.path.join(self.models_path, "deepchem_eval")
        self.path_final = os.path.join(self.models_path, "deepchem_final")

        self.atom_features = Feature(shape=(None, 75), name="atom")
        self.degree_slice = Feature(
            shape=(None, 2), dtype=tf.int32, name="dgr_slice")
        self.membership = Feature(
            shape=(None,), dtype=tf.int32, name="membership")
        self.label = Label(shape=(None, 128), name="label_sign3")
        self.weights = Weights(shape=(None, 1), name="weights_sign3")
        self.deg_adjs = []
        for i in range(0, 10 + 1):
            deg_adj = Feature(shape=(None, i + 1),
                              dtype=tf.int32, name="dgr_adj_" + str(i))
            print(deg_adj.name)
            self.deg_adjs.append(deg_adj)

    def _default_generator(self, dataset_iter, epochs=1, batch_size=1000, label=None,
                           weights=None, atom_features=None, degree_slice=None,
                           membership=None, deg_adjs=None):
        featurizer = ConvMolFeaturizer()
        for epoch in range(epochs):
            for x, y in dataset_iter():
                d = {}

                new_sml = np.array([sml[0] for sml in x])
                smile_pd = pd.Series(new_sml).to_frame()

                ds_smiles, indices = featurize_smiles_df(
                    smile_pd, featurizer, 0)
                w = np.ones((len(indices), 1))
                X_b, y_b, w_b, ids_b = pad_batch(batch_size, ds_smiles, y[
                                                 indices], w[indices], new_sml)

                d[label] = y_b

                d[weights] = w_b

                print(ds_smiles.shape, X_b.shape)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                d[atom_features] = multiConvMol.get_atom_features()
                d[degree_slice] = multiConvMol.deg_slice
                d[membership] = multiConvMol.membership
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
                yield d

    def _default_generator_predict(self, smiles, epochs=1, batch_size=1000, label=None,
                                   weights=None, atom_features=None, degree_slice=None,
                                   membership=None, deg_adjs=None):
        featurizer = ConvMolFeaturizer()
        for epoch in range(epochs):
            for i in range(0, len(smiles), batch_size):
                chunk = slice(i, i + batch_size)
                x = smiles[chunk]
                d = {}

                new_sml = np.array([sml[0] for sml in x])
                smile_pd = pd.Series(new_sml).to_frame()

                ds_smiles, indices = featurize_smiles_df(
                    smile_pd, featurizer, 0)
                w = np.ones((len(indices), 1))
                X_b, y_b, w_b, ids_b = pad_batch(
                    batch_size, ds_smiles, None, w[indices], new_sml)

                d[weights] = w_b

                print(ds_smiles.shape, X_b.shape)
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                d[atom_features] = multiConvMol.get_atom_features()
                d[degree_slice] = multiConvMol.deg_slice
                d[membership] = multiConvMol.membership
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
                yield d

    def _layer_reference(self, model, layer):
        if isinstance(layer, list):
            return [model.layers[x.name] for x in layer]
        print(layer.name)
        return model.layers[layer.name]

    def fit(self, data_path, evaluate=True):
        """Take data .h5 and learn an encoder.

        Args:
            data_path(string): a path to .h5 file.
        """
        self.data_path = data_path

        config = tf.ConfigProto(intra_op_parallelism_threads=self.cpu,
                                inter_op_parallelism_threads=self.cpu,
                                allow_soft_placement=True,
                                device_count={'CPU': self.cpu})

        if not os.path.isfile(self.data_path) or self.data_path[-3:] != '.h5':
            raise Exception("Input data needs to be a H5 file")

        self.traintest_file = os.path.join(self.models_path, 'traintest.h5')

        with h5py.File(data_path, 'r') as hf:
            if 'x' not in hf.keys():
                raise Exception(
                    "Input data file needs to have a dataset called 'x'")

        if not os.path.isfile(self.traintest_file):
            Traintest.split_h5_blocks(self.data_path, self.traintest_file, split_names=[
                'train', 'test'], split_fractions=[.8, .2], datasets=['x', 'y'])

        model_final = TensorGraph(tensorboard=True,
                                  batch_size=self.batch_size,
                                  learning_rate=self.learning_rate,
                                  use_queue=False,
                                  model_dir=self.path_final,
                                  configproto=config)

        gc1 = GraphConv(
            64,
            activation_fn=tf.nn.tanh,
            in_layers=[self.atom_features, self.degree_slice, self.membership] + self.deg_adjs)
        batch_norm1 = BatchNorm(in_layers=[gc1])
        gp1 = GraphPool(
            in_layers=[batch_norm1, self.degree_slice, self.membership] + self.deg_adjs)
        gc2 = GraphConv(
            64,
            activation_fn=tf.nn.tanh,
            in_layers=[gp1, self.degree_slice, self.membership] + self.deg_adjs)
        batch_norm2 = BatchNorm(in_layers=[gc2])
        gp2 = GraphPool(
            in_layers=[batch_norm2, self.degree_slice, self.membership] + self.deg_adjs)
        dense = Dense(out_channels=128,
                      activation_fn=tf.nn.tanh, in_layers=[gp2])
        batch_norm3 = BatchNorm(in_layers=[dense])
        readout = GraphGather(
            batch_size=self.batch_size,
            activation_fn=tf.nn.tanh,
            in_layers=[batch_norm3, self.degree_slice, self.membership] + self.deg_adjs)

        classification = Dense(
            out_channels=128, activation_fn=None, in_layers=[readout])

        if evaluate:

            model_eval = TensorGraph(tensorboard=True,
                                     batch_size=self.batch_size,
                                     learning_rate=self.learning_rate,
                                     use_queue=False,
                                     model_dir=self.path_eval,
                                     configproto=config)

            model_eval.add_output(classification)

            cost = L2Loss(
                in_layers=[self.label, classification], name="L2Loss")

            loss = WeightedError(in_layers=[cost, self.weights])
            model_eval.set_loss(loss)

            results = {}
            (x_shape, y_shape), dtypes, iterator_train_evaluate = Traintest.generator_fn(
                self.traintest_file, "train", self.batch_size)

            (x_shape, y_shape), dtypes, iterator_test_evaluate = Traintest.generator_fn(
                self.traintest_file, "test", self.batch_size)

            gene = self._default_generator(iterator_train_evaluate, epochs=self.epochs, batch_size=self.batch_size,
                                           label=self.label, weights=self.weights, atom_features=self.atom_features, degree_slice=self.degree_slice,
                                           membership=self.membership, deg_adjs=self.deg_adjs)
            model_eval.fit_generator(gene)

            gene = self._default_generator(iterator_train_evaluate, epochs=self.epochs, batch_size=self.batch_size,
                                           label=self.label, weights=self.weights, atom_features=self.atom_features, degree_slice=self.degree_slice,
                                           membership=self.membership, deg_adjs=self.deg_adjs)

            train_predictions = model_eval.predict_on_generator(generator=gene)

            metric = dc.metrics.Metric(
                dc.metrics.mean_squared_error, np.mean, mode="regression")

            train_data = Traintest(self.traintest_file, "train")
            train_data.open()
            ytrue = train_data.get_all_y()
            train_data.close()
            train_predictions = train_predictions[:ytrue.shape[0]]
            score_train = metric.compute_metric(
                ytrue, train_predictions, np.ones((train_predictions.shape[0], 128)))

            results["score_train"] = float(score_train)

            gene = self._default_generator(iterator_test_evaluate, epochs=self.epochs, batch_size=self.batch_size,
                                           label=self.label, weights=self.weights, atom_features=self.atom_features, degree_slice=self.degree_slice,
                                           membership=self.membership, deg_adjs=self.deg_adjs)

            test_predictions = model_eval.predict_on_generator(generator=gene)

            metric = dc.metrics.Metric(
                dc.metrics.mean_squared_error, np.mean, mode="regression")

            test_data = Traintest(self.traintest_file, "test")
            test_data.open()
            ytrue = test_data.get_all_y()
            test_data.close()
            test_predictions = test_predictions[:ytrue.shape[0]]
            score_test = metric.compute_metric(
                ytrue, test_predictions, np.ones((test_predictions.shape[0], 128)))
            results["score_test"] = float(score_test)
            print(results)
            with open(os.path.join(self.models_path, "results.json"), 'w') as f:
                json.dump(results, f)

        model_final.add_output(classification)

        cost = L2Loss(in_layers=[self.label, classification], name="L2Loss")

        loss = WeightedError(in_layers=[cost, self.weights])
        model_final.set_loss(loss)

        (x_shape, y_shape), dtypes, iterator_train = Traintest.generator_fn(
            self.data_path, None, self.batch_size)

        gene = self._default_generator(iterator_train, epochs=self.epochs, batch_size=self.batch_size,
                                       label=self.label, weights=self.weights, atom_features=self.atom_features, degree_slice=self.degree_slice,
                                       membership=self.membership, deg_adjs=self.deg_adjs)
        model_final.fit_generator(gene)

        model_final.save()

    def predict_from_smiles(self, smiles, dest_file):
        """Given SMILES generate sign0 and predict sign3.

        Args:
            smiles(list): A list of SMILES strings. We assume the user already
                standardized the SMILES string.
            dest_file(str): File where to save the predictions.
        Returns:
            pred_s3(DataSignature): The predicted signatures as DataSignature
                object.
        """

        config = tf.ConfigProto(intra_op_parallelism_threads=self.cpu,
                                inter_op_parallelism_threads=self.cpu,
                                allow_soft_placement=True,
                                device_count={'CPU': self.cpu})

        model2 = TensorGraph.load_from_dir(
            self.path_final)  # Load the saved model

        model2.configproto = config

        for layer in model2.layers:
            print(layer)

        label = self._layer_reference(model2, self.label)
        weights = self._layer_reference(model2, self.weights)
        atom_features = self._layer_reference(model2, self.atom_features)
        degree_slice = self._layer_reference(model2, self.degree_slice)
        membership = self._layer_reference(model2, self.membership)
        deg_adjs = self._layer_reference(model2, self.deg_adjs)

        gene = self._default_generator_predict(smiles, epochs=self.epochs, batch_size=self.batch_size,
                                               label=label, weights=weights, atom_features=atom_features, degree_slice=degree_slice,
                                               membership=membership, deg_adjs=deg_adjs)

        predictions = model2.predict_on_generator(generator=gene)

        # we return a simple DataSignature object (basic HDF5 access)
        pred_s3 = DataSignature(dest_file)
        with h5py.File(dest_file, "w") as results:
            # initialize V (with NaN in case of failing rdkit) and smiles keys
            results.create_dataset('keys', data=np.array(
                smiles, DataSignature.string_dtype()))
            results.create_dataset('V', (len(smiles), 128), dtype=np.float32)

            results["V"][:] = predictions[:len(smiles)]

        return pred_s3
