The Chemical Checker
====================

The Chemical Checker (CC) is a data-driven resource of small molecule
bioactivity data. The main goal of the CC is to express data in a format
that can be used off-the-shelf in daily computational drug discovery
tasks. The resource is organized in **5 levels** of increasing
complexity, ranging from the chemical properties of the compounds to
their clinical outcomes. In between, we consider targets, off-targets,
perturbed biological networks and several cell-based assays, including
gene expression, growth inhibition, and morphological profiles. The CC
is different to other integrative compounds database in almost every
aspect. The classical, relational representation of the data is
surpassed here by a less explicit, more machine-learning-friendly
abstraction of the data.

The CC resource is ever-growing and maintained by the 
`Structural Bioinformatics & Network Biology Laboratory`_ 
at the Institute for
Research in Biomedicine (`IRB Barcelona`_). Should you have any
questions, please send an email to miquel.duran@irbbarcelona.org or
patrick.aloy@irbbarcelona.org.

This project was first presented to the scientific community in the
following paper:  

    Duran-Frigola M, et al
    "**Extending the small-molecule similarity principle to all levels of biology with the Chemical Checker.**"
    Nature Biotechnology (2020) [`link`_]

and has since produced a number of `related publications`_.

.. note::
    For an overview of the CC universe please visit `bioactivitysignatures.org`_

.. _Structural Bioinformatics & Network Biology Laboratory: https://sbnb.irbbarcelona.org/
.. _IRB Barcelona: https://www.irbbarcelona.org/en
.. _related publications: https://www.bioactivitysignatures.org/publications.html
.. _link: https://www.nature.com/articles/s41587-020-0502-7
.. _BioactivitySignatures.org: https://www.bioactivitysignatures.org/


Source data and datasets
------------------------

The CC is built from public bioactivity data. We are committed to
updating the resource **every 6 months** (versions named accordingly,
e.g. ``chemical_checker_2019_01``). New datasets may be incorporated
upon request.

The basic data unit of the CC is the *dataset*. There are 5 data
*levels* (``A`` Chemistry, ``B`` Targets, ``C`` Networks, ``D`` Cells
and ``E`` Clinics) and, in turn, each level is divided into 5 sublevels
or *coordinates* (``A1``-``E5``). Each dataset belongs to one and only
one of the 25 coordinates, and each coordinate can have a finite number
of datasets (e.g. ``A1.001``), one of which is selected as being
*exemplary*.

The CC is a chemistry-first biomedical resource and, as such, it
contains several predefined compound collections that are of interest to
drug discoverers, including approved drugs, natural products, and
commercial screening libraries.


Signaturization of the data
---------------------------

The main task of the CC is to convert raw data into formats that are
suitable inputs for machine-learning toolkits such as `scikit-learn`_.

Accordingly, the backbone pipeline of the CC is devoted to processing
every dataset and converting it to a series of formats that may be
readily useful for machine learning. The main assets of the CC are the
so-called *CC signatures*:

+-------------+-------------+-------------+-------------+-------------+
| Signature   | Abbreviation| Description | Advantages  |Disadvantages|
+=============+=============+=============+=============+=============+
| Type 0      | ``sign0``   | Raw dataset | Explicit    | Possibly    |
|             |             | data,       | data.       | sparse,     |
|             |             | expressed   |             | het         |
|             |             | in a matrix |             | erogeneous, |
|             |             | format.     |             | u           |
|             |             |             |             | nprocessed. |
+-------------+-------------+-------------+-------------+-------------+
| Type 1      | ``sign1``   | PCA/LSI     | Biological  | Variables   |
|             |             | projections | signatures  | dimensions, |
|             |             | of the      | of this     | they may    |
|             |             | data,       | type can be | still be    |
|             |             | accounting  | obtained by | sparse.     |
|             |             | for 90% of  | simple      |             |
|             |             | the data.   | projection. |             |
|             |             |             | Easy to     |             |
|             |             |             | compute and |             |
|             |             |             | require no  |             |
|             |             |             | f           |             |
|             |             |             | ine-tuning. |             |
+-------------+-------------+-------------+-------------+-------------+
| Type 2      | ``sign2``   | Networ      | Fixed       | Information |
|             |             | k-embedding | -length,    | leak due to |
|             |             | of the      | usually     | similarity  |
|             |             | similarity  | acceptably  | measures.   |
|             |             | network.    | short.      | Hype        |
|             |             |             | Suitable    | r-parameter |
|             |             |             | for machine | tunning.    |
|             |             |             | learning.   |             |
|             |             |             | Capture     |             |
|             |             |             | global      |             |
|             |             |             | properties  |             |
|             |             |             | of the      |             |
|             |             |             | similarity  |             |
|             |             |             | network.    |             |
+-------------+-------------+-------------+-------------+-------------+
| Type 3      | ``sign3``   | Networ      | Fixed       | Possibly    |
|             |             | k-embedding | dimension   | very noisy, |
|             |             | of the      | and         | hence       |
|             |             | inferred    | available   | useless,    |
|             |             | similarity  | for *any*   | especially  |
|             |             | network.    | molecule.   | for         |
|             |             |             |             | low-data    |
|             |             |             |             | datasets.   |
+-------------+-------------+-------------+-------------+-------------+

.. note::
    A `Signaturizer`_ module for direct molecule signaturization is also available.

.. _scikit-learn: https://scikit-learn.org/
.. _Signaturizer: http://gitlabsbnb.irbbarcelona.org/packages/signaturizer