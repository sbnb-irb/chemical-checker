Signaturization
===============

The main feature of the CC is the automatic conversion of virtually any compound-related data into a standard format ready-to-be-used by machine learning algorithms. All CC data is stored as ``HDF5`` files following a folder structure organized by datasets (described in :mod:`chemicalchecker.core.chemcheck`).

The central type of data is the **signature** (one numerical vector per molecule). There are 5 types of signatures:

    * **Signatures type 0** :class:`~chemicalchecker.core.sign0`: A sufficiently-processed version of the raw data. They usually show explicit knowledge, which enables connectivity and interpretation.
    * **Signatures type 1** :class:`~chemicalchecker.core.sign1`: A mildly compressed (usually latent) representation of the signatures, with a dimensionality that typically retains 90% of the original variance. They keep most of the complexity of the original data and they can be used for similarity calculations.
    * **Signatures type 2** :class:`~chemicalchecker.core.sign2`: Network embedding of the similarity matrix derived from signatures type 1. These signatures have a fixed length (e.g. 128-d), which is convenient for machine learning, and capture both explicit *and* implicit similarity relationships in the data.
    * **Signatures type 3** :class:`~chemicalchecker.core.sign3`: Fixed-length (e.g. 128-d) representation of the data, capturing *and* inferring the original (signature type 1) similarity of the data. Signatures type 3 are available for any molecule in the Chemical Checker universe (~1 Million) and have a confidence measure assigned to them.
    * **Signatures type 4** :class:`~chemicalchecker.core.sign4`: Fixed-length (e.g. 128-d) representation of the data, inferring the signature type 3. Signatures type 4 are available for any molecule of interest and have a confidence measure assigned to them. Are convinently generated with the *signaturizer* tool.

Besides, there are other auxiliary types of data that can be computed for each of the signature types. 

    * **Nearest neighbors** :class:`~chemicalchecker.core.neig`: Nearest neighbors search result. Currently, we consider the 1000-nearest neighbors, which is more than sufficient in any realistic scenario.
    * **Clusters** :class:`~chemicalchecker.core.clus`: Centroids and labels of a k-means clustering.
    * **2D Projections** :class:`~chemicalchecker.core.proj`: Various (t-SNE, UMAP, PCA) 2D projections of the data.

Signature characteristics
'''''''''''''''''''''''''

Every CC signature must have:

* A matrix of data ``V``.
* Keys (typically InChIKeys) ``keys``.
* A ``fit`` method to be run at production time.
* A ``predict`` method to be run for out-of-production (new) data.

Below we detail the characteristics of each signature type and the algorithms behind.

All signature classes inherit from two base-classes that enforce shared behavior:

   *. :class:`~chemicalchecker.core.signature_base.BaseSignature`: uses
      low level ``HDF5`` to optimize access to signature persistence level (e.g.
      the file ``/root/full/A/A1/A1.001/sign2/sign2.h5``)
   *. :class:`~chemicalchecker.core.signature_data.DataSignature`: implements
      sanity checks, internal signature directory structure, fit and predict
      hooks, etc...

Signatures type 0
'''''''''''''''''

These are the raw signatures that enter the CC pipeline. The input of these signatures can be:

* *Sparse*: A ``pairs`` vector (e.g. molecule-target pairs), optionally with weights.
* *Dense*: An ``X`` matrix with ``keys`` (e.g. molecules) and ``features`` (e.g. cell-lines).

Signatures type 0 are minimally modified. We only apply the following modifications:

* *Imputation*: for *dense* inputs, ``NA`` values are median-imputed.
* *Aggregation*: In case some keys are duplicated (for instance, at fit and predict time), user can choose to keep the first instance, the last instance, or an average of the data.

See implementation details in :mod:`~chemicalchecker.core.sign0`

Signatures type 1
'''''''''''''''''

These signatures are processed versions of the *experimental* data available in the CC and can be used for similarity measures. They have variable dimensionality, depending on the CC space.

The CC provides an automated pipeline for producing signatures type 1. We allow some flexibility and the user can choose to apply the following procedures:

* *Latent representation*: For originally sparse matrices, this is TF-IDF Latent-semantic indexing (LSI). For dense matrices, this corresponds to a PCA (optionally, preceded by a robust median-based scaling). By default, we keep 90% of the variance.
* *Outlier removal*: Outlier keys (molecules) can be removed based on the isolation forest algorithm.
* *Metric learning*: A shallow siamese network can be trained to learn a latent representation of the space that has a good distance distribution. Metric learning requires similar/dissimilar cases (triplets); we offer two options:

  * *Unsupervised*: Triplets are drawn from the signature itself.
  * *Semi-supervised*: Triplets are drawn from other CC signatures. This helps integrate/contextualize the signature in question within the framework of the CC.

See implementation details in :mod:`~chemicalchecker.core.sign1`

Signatures type 2
'''''''''''''''''

These signatures are mostly used for internal machine-learning procedures, as they have a convenient fixed-length format.

Signatures type 2 are produced with two steps:

   1.  Construction of a similarity network using signatures type 1.
   2.  Network embedding (using :mod:`~chemicalchecker.tool.node2vec`).

See implementation details in :mod:`~chemicalchecker.core.sign2`

Signatures type 3
'''''''''''''''''

These signatures are fixed-length vectors available for all molecules of the Chemical Checker universe (~1 Million). Thus, they are mostly *inferred* properties.

To learn signatures type 3:

* Triplets are sampled from type 1 similarities
* Signatures type 2 across the CC are used as input for a deep siamese neural network. Thus, 25 fixed-length vectors are stacked.
* A signature-dropout (subsampling) procedure is applied to ensure that the data seen in the training set are *realistic*, meaning that the signature coverage resembles the coverage available for those molecules that do *not* have data available for the CC space in question.

Applicability
-------------

An ``applicability`` score represent how "confident" we can be about a 
specific signature type 3. This accuracy estimate is based on:

* *Distance*: signatures that are close to training-set signatures are, in principle, closer to the
  applicability domain. We measure distances in an unsupervised way (i.e. average distance
  to 5/25 nearest-neighbors) and in a supervised way by means of a random forest regressor
  trained on signatures as features and prediction accuracy (correlation) as dependent variable. In
  addition, we devised a measure of ‘intensity’, defined as the mean absolute deviation of the
  signatures to the average (null) signature observed in the training set.
* *Robustness*: The signature-dropout procedure presented above can be applied at prediction
  time to get an estimate of the robustness of the prediction. For each molecule, we generated 10
  dropped-out inputs, thereby obtaining an ensemble of predictions. Small standard deviations
  over these predictions indicate a robust output.
* *Expectancy a priori*: We calculated the accuracy that is expected given the input signatures
  available for a particular molecule. Some CC signature types are highly predictive for others;
  thus, having these informative signatures at hand will in principle favor reliable predictions. This
  prior expectancy was calculated by fitting a random forest classifier having 25
  absence/presence features as covariates and prediction accuracy as outcome.

See implementation details in :mod:`~chemicalchecker.core.sign3`


Signatures type 4
'''''''''''''''''

These signatures are fixed-length vectors available for *any* molecule of interest. They are completely *inferred*.

To learn signatures type 4:

* A DNN for multi-output regression problem is trained to predict signatures type 3 from a simple ECFP4 (Morgan fingerprint) representation.
* These signatures correspond to *signaturizer* produced signatures