Datasets
========

In the CC nomenclature, a dataset is determined by:

1. One coordinate.
2. One (typically) or multiple (eventually) sources having the same type
   of (mergeable) data.
3. A `pre-processing procedure` yielding signatures type 0.

Levels, coordinates and datasets
--------------------------------

The CC is divided into five **levels** of increasing complexity:

===== ========= ===================================================
Level Name      Description
===== ========= ===================================================
``A`` Chemistry Chemical properties of the compounds.
``B`` Targets   Chemical-protein interactions.
``C`` Networks  Higher-order effects of small molecules.
``D`` Cells     Readouts of compound cell-based assays.
``C`` Clinics   Clinical data of drugs and environmental chemicals.
===== ========= ===================================================

In turn, each level is divided into 5 sublevels or **coordinates**
representing different aspects of the data. Each sublevel has an
*exemplary* dataset, as described below:

+-----------------------+-----------------------+-----------------------+
| Coordinate            | Name                  | Description           |
+=======================+=======================+=======================+
| ``A1``                | 2D fingerprints       | Binary representation |
|                       |                       | of the 2D structure   |
|                       |                       | of a molecule. The    |
|                       |                       | neighbourhood of      |
|                       |                       | every atom is encoded |
|                       |                       | using circular        |
|                       |                       | topology hashing.     |
+-----------------------+-----------------------+-----------------------+
| ``A2``                | 3D fingerprints       | Similar to ``A1``,    |
|                       |                       | the 3D structures of  |
|                       |                       | the three best        |
|                       |                       | conformers after      |
|                       |                       | energy minimization   |
|                       |                       | are hashed into a     |
|                       |                       | binary representation |
|                       |                       | without the need for  |
|                       |                       | structural alignment. |
+-----------------------+-----------------------+-----------------------+
| ``A3``                | Scaffolds             | Largest molecular     |
|                       |                       | scaffold (usually a   |
|                       |                       | ring system)          |
|                       |                       | remaining after       |
|                       |                       | applying Murcko’s     |
|                       |                       | pruning rules.        |
|                       |                       | Additionally, we keep |
|                       |                       | the corresponding     |
|                       |                       | framework, i.e. a     |
|                       |                       | version of the        |
|                       |                       | scaffold where all    |
|                       |                       | atoms are carbons and |
|                       |                       | all bonds are single. |
|                       |                       | The scaffold and the  |
|                       |                       | framework are encoded |
|                       |                       | with path-based       |
|                       |                       | 1024-bit              |
|                       |                       | fingerprints,         |
|                       |                       | suitable for          |
|                       |                       | capturing             |
|                       |                       | substructures in      |
|                       |                       | similarity searches.  |
+-----------------------+-----------------------+-----------------------+
| ``A4``                | Structural keys       | 166 functional groups |
|                       |                       | and substructures     |
|                       |                       | widely accepted by    |
|                       |                       | medicinal chemists    |
|                       |                       | (MACCS keys).         |
+-----------------------+-----------------------+-----------------------+
| ``A5``                | Physicochemistry      | Physicochemical       |
|                       |                       | properties such as    |
|                       |                       | molecular weight,     |
|                       |                       | logP, and             |
|                       |                       | refractivity. Number  |
|                       |                       | of hydrogen-bond      |
|                       |                       | donors and acceptors, |
|                       |                       | rings, etc.           |
|                       |                       | Drug-likeness         |
|                       |                       | measurements          |
|                       |                       | e.g. number of        |
|                       |                       | structural alerts,    |
|                       |                       | Lipinski’s rule-of-5  |
|                       |                       | violations or         |
|                       |                       | chemical beauty       |
|                       |                       | (QED).                |
+-----------------------+-----------------------+-----------------------+
| ``B1``                | Mechanism of action   | Drug targets with     |
|                       |                       | known pharmacological |
|                       |                       | action and modes      |
|                       |                       | (agonist, antagonist, |
|                       |                       | etc.).                |
+-----------------------+-----------------------+-----------------------+
| ``B2``                | Metabolic genes       | Drug metabolizing     |
|                       |                       | enzymes,              |
|                       |                       | transporters, and     |
|                       |                       | carriers.             |
+-----------------------+-----------------------+-----------------------+
| ``B3``                | Crystals              | Small molecules       |
|                       |                       | co-crystalized with   |
|                       |                       | protein chains. Data  |
|                       |                       | is organized          |
|                       |                       | according to the      |
|                       |                       | structural families   |
|                       |                       | of the protein        |
|                       |                       | chains.               |
+-----------------------+-----------------------+-----------------------+
| ``B4``                | Binding               | Compound–protein      |
|                       |                       | binding data          |
|                       |                       | available in major    |
|                       |                       | public chemogenomics  |
|                       |                       | databases. Data       |
|                       |                       | mainly comes from     |
|                       |                       | academic publications |
|                       |                       | and patents. Only     |
|                       |                       | binding affinities    |
|                       |                       | below a               |
|                       |                       | class-specific        |
|                       |                       | threshold are kept    |
|                       |                       | (kinases ≤ 30 nM,     |
|                       |                       | GPCRs ≤ 100 nM,       |
|                       |                       | nuclear receptors ≤   |
|                       |                       | 100 nM, ion channels  |
|                       |                       | ≤ 10 uM and others ≤  |
|                       |                       | 1 uM).                |
+-----------------------+-----------------------+-----------------------+
| ``B5``                | HTS bioassays         | Hits from screening   |
|                       |                       | campaigns against     |
|                       |                       | protein targets       |
|                       |                       | (mainly confirmatory  |
|                       |                       | functional assays     |
|                       |                       | below 10 uM).         |
+-----------------------+-----------------------+-----------------------+
| ``C1``                | Biological roles      | Ontology terms        |
|                       |                       | associated with small |
|                       |                       | molecules with        |
|                       |                       | recognized biological |
|                       |                       | roles, such as known  |
|                       |                       | drugs, metabolites    |
|                       |                       | and other natural     |
|                       |                       | products.             |
+-----------------------+-----------------------+-----------------------+
| ``C2``                | Metabolic network     | Curated               |
|                       |                       | reconstruction of     |
|                       |                       | human metabolism,     |
|                       |                       | containing            |
|                       |                       | metabolites and       |
|                       |                       | reactions. Data is    |
|                       |                       | represented as a      |
|                       |                       | network where nodes   |
|                       |                       | are metabolites and   |
|                       |                       | edges connect         |
|                       |                       | substrates and        |
|                       |                       | products of           |
|                       |                       | reactions.            |
+-----------------------+-----------------------+-----------------------+
| ``C3``                | Canonical pathways    | Canonical pathways    |
|                       |                       | related to the known  |
|                       |                       | receptors of          |
|                       |                       | compounds (as         |
|                       |                       | recorded in ``B4``).  |
|                       |                       | Pathways are assigned |
|                       |                       | via a                 |
|                       |                       | guilt-by-association  |
|                       |                       | approach, i.e. a      |
|                       |                       | molecule is related   |
|                       |                       | to a pathway if at    |
|                       |                       | least one of the      |
|                       |                       | molecule targets is a |
|                       |                       | member of it.         |
+-----------------------+-----------------------+-----------------------+
| ``C4``                | Biological processes  | Similar to ``C3``,    |
|                       |                       | biological processes  |
|                       |                       | from the gene         |
|                       |                       | ontology are          |
|                       |                       | associated with       |
|                       |                       | compounds via a       |
|                       |                       | guilt-by-association  |
|                       |                       | approach from ``B4``  |
|                       |                       | data. All parent      |
|                       |                       | terms are kept, from  |
|                       |                       | the leaves of the     |
|                       |                       | ontology to its root. |
+-----------------------+-----------------------+-----------------------+
| ``C5``                | Interactomes          | Neighborhoods of      |
|                       |                       | ``B4`` targets are    |
|                       |                       | collected by          |
|                       |                       | inspecting several    |
|                       |                       | large protein-protein |
|                       |                       | interaction networks. |
|                       |                       | A random-walk         |
|                       |                       | algorithm is used to  |
|                       |                       | obtain a robust       |
|                       |                       | measure of            |
|                       |                       | ‘proximity’ in the    |
|                       |                       | network.              |
+-----------------------+-----------------------+-----------------------+
| ``D1``                | Gene expression       | Transcriptional       |
|                       |                       | response of cell      |
|                       |                       | lines upon exposure   |
|                       |                       | to small molecules. A |
|                       |                       | well-documented       |
|                       |                       | reference dataset of  |
|                       |                       | gene expression       |
|                       |                       | profiles is used to   |
|                       |                       | map all compound      |
|                       |                       | profiles using a      |
|                       |                       | two-sided gene set    |
|                       |                       | enrichment analysis.  |
+-----------------------+-----------------------+-----------------------+
| ``D2``                | Cancer cell lines     | Small molecule        |
|                       |                       | sensitivity data      |
|                       |                       | (GI50) of a panel of  |
|                       |                       | 60 cancer cell lines. |
+-----------------------+-----------------------+-----------------------+
| ``D3``                | Chemical genetics     | Growth inhibition     |
|                       |                       | profiles in a panel   |
|                       |                       | of ~300 yeast         |
|                       |                       | mutants. Data are     |
|                       |                       | combined with yeast   |
|                       |                       | genetic interaction   |
|                       |                       | data so that          |
|                       |                       | compounds can be      |
|                       |                       | assimilated to        |
|                       |                       | genetic alterations   |
|                       |                       | when they have        |
|                       |                       | similar profiles.     |
+-----------------------+-----------------------+-----------------------+
| ``D4``                | Morphology            | Changes in U-2 OS     |
|                       |                       | cell morphology       |
|                       |                       | measured after        |
|                       |                       | compound treatment    |
|                       |                       | using a               |
|                       |                       | mu                    |
|                       |                       | ltiplexed-cytological |
|                       |                       | cell painting assay.  |
|                       |                       | 812 morphology        |
|                       |                       | features are recorded |
|                       |                       | via automated         |
|                       |                       | microscopy and image  |
|                       |                       | analysis.             |
+-----------------------+-----------------------+-----------------------+
| ``D5``                | Cell bioassays        | Small molecule cell   |
|                       |                       | bioassays reported in |
|                       |                       | ChEMBL, mainly growth |
|                       |                       | and proliferation     |
|                       |                       | measurements found in |
|                       |                       | the literature.       |
+-----------------------+-----------------------+-----------------------+
| ``E1``                | Therapeutic areas     | Anatomical            |
|                       |                       | Therapeutic Chemical  |
|                       |                       | (ATC) codes of drugs. |
|                       |                       | All ATC levels are    |
|                       |                       | considered.           |
+-----------------------+-----------------------+-----------------------+
| ``E2``                | Indications           | Indications of        |
|                       |                       | approved drugs and    |
|                       |                       | drugs in clinical     |
|                       |                       | trials. A controlled  |
|                       |                       | medical vocabulary is |
|                       |                       | used.                 |
+-----------------------+-----------------------+-----------------------+
| ``E3``                | Side effects          | Side effects          |
|                       |                       | extracted from drug   |
|                       |                       | package inserts via   |
|                       |                       | text-mining           |
|                       |                       | techniques.           |
+-----------------------+-----------------------+-----------------------+
| ``E4``                | Disease phenotypes    | Manually curated      |
|                       |                       | relationships between |
|                       |                       | chemicals and         |
|                       |                       | diseases. Chemicals   |
|                       |                       | include drug          |
|                       |                       | molecules and         |
|                       |                       | environmental         |
|                       |                       | substances, among     |
|                       |                       | others.               |
+-----------------------+-----------------------+-----------------------+
| ``E5``                | Drug-drug             | Changes in the effect |
|                       | interactions          | of a drug when it is  |
|                       |                       | taken together with a |
|                       |                       | second drug.          |
|                       |                       | Drug-drug             |
|                       |                       | interactions may      |
|                       |                       | alter                 |
|                       |                       | pharmacokinetics      |
|                       |                       | and/or cause side     |
|                       |                       | effects.              |
+-----------------------+-----------------------+-----------------------+

Each of the coordinates can contain an arbitrary number of **datasets** 
with increasing number (e.g. ``A1.001``).

Dataset characteristics
-----------------------

This is how we define a dataset:

+-----------------------+-----------------------+-----------------------+
| Column                | Values                | Description           |
+=======================+=======================+=======================+
| Code                  | e.g.\ ``A1.001``      | Identifier of the     |
|                       |                       | dataset.              |
+-----------------------+-----------------------+-----------------------+
| Level                 | e.g. ``A``            | The CC level.         |
+-----------------------+-----------------------+-----------------------+
| Coordinate            | e.g.\ ``A1``          | Coordinates in the CC |
|                       |                       | organization.         |
+-----------------------+-----------------------+-----------------------+
| Name                  | 2D fingerprints       | Display, short-name   |
|                       |                       | of the dataset.       |
+-----------------------+-----------------------+-----------------------+
| Technical name        | 1024-bit Morgan       | A more technical name |
|                       | fingerprints          | for the dataset,      |
|                       |                       | suitable for          |
|                       |                       | chemo                 |
|                       |                       | -/bio-informaticians. |
+-----------------------+-----------------------+-----------------------+
| Description           | 2D fingerprints are…  | This field contains a |
|                       |                       | long description of   |
|                       |                       | the dataset. It is    |
|                       |                       | important that the    |
|                       |                       | curator outlines here |
|                       |                       | the importance of the |
|                       |                       | dataset, why did      |
|                       |                       | he/she make the       |
|                       |                       | decision to include   |
|                       |                       | it, and what are the  |
|                       |                       | scenarios where this  |
|                       |                       | dataset may be        |
|                       |                       | useful.               |
+-----------------------+-----------------------+-----------------------+
| Unknowns              | ``True``/``False``    | Does the dataset      |
|                       |                       | contain known/unknown |
|                       |                       | data? Binding data    |
|                       |                       | from chemogenomics    |
|                       |                       | datasets, for         |
|                       |                       | example, are          |
|                       |                       | positive-unlabeled,   |
|                       |                       | so they do contain    |
|                       |                       | unknowns. Conversely, |
|                       |                       | chemical fingerprints |
|                       |                       | or gene expression    |
|                       |                       | data do not contain   |
|                       |                       | unknowns.             |
+-----------------------+-----------------------+-----------------------+
| Discrete              | ``True``/``False``    | The type of data that |
|                       |                       | ultimately expresses  |
|                       |                       | de dataset, after the |
|                       |                       | pre-processing.       |
|                       |                       | Categorical variables |
|                       |                       | are not allowed; they |
|                       |                       | must be converted to  |
|                       |                       | one-hot encoding or   |
|                       |                       | binarized. Mixed      |
|                       |                       | variables are not     |
|                       |                       | allowed, either.      |
+-----------------------+-----------------------+-----------------------+
| Keys                  | e.g. ``CPD`` (we use  | In the core CC        |
|                       | @afernandez           | database, most of the |
|                       | ``Bioteque``          | times this field will |
|                       | nomenclature). Can be | correspond to         |
|                       | ``NULL``.             | ``CPD``, as the CC is |
|                       |                       | centred on small      |
|                       |                       | molecules. It only    |
|                       |                       | makes sense to have   |
|                       |                       | keys of different     |
|                       |                       | types when we do      |
|                       |                       | connectivity          |
|                       |                       | attempts, that is,    |
|                       |                       | for example, when     |
|                       |                       | mapping disease gene  |
|                       |                       | expression            |
|                       |                       | signatures.           |
+-----------------------+-----------------------+-----------------------+
| Features              | e.g. ``GEN`` (we use  | When features         |
|                       | ``Bioteque``          | correspond to         |
|                       | nomenclature). Can be | explicit knowledge,   |
|                       | ``NULL``.             | such as proteins,     |
|                       |                       | gene ontology         |
|                       |                       | processes, or         |
|                       |                       | indications, we       |
|                       |                       | express with this     |
|                       |                       | field the type of     |
|                       |                       | biological entities.  |
|                       |                       | It is not allowed to  |
|                       |                       | mix different feature |
|                       |                       | types. Features can,  |
|                       |                       | however, have no      |
|                       |                       | type, typically when  |
|                       |                       | they come from a      |
|                       |                       | heavily-processed     |
|                       |                       | dataset, such as      |
|                       |                       | gene-expression data. |
|                       |                       | Even if we use        |
|                       |                       | ``Bioteque``          |
|                       |                       | nomenclature to the   |
|                       |                       | define the type of    |
|                       |                       | biological data, it   |
|                       |                       | is not mandatory that |
|                       |                       | the vocabularies are  |
|                       |                       | the ones used by the  |
|                       |                       | ``Bioteque``; for     |
|                       |                       | example, I can use    |
|                       |                       | non-human UniProt     |
|                       |                       | ACs, if I deem it     |
|                       |                       | necessary.            |
+-----------------------+-----------------------+-----------------------+
| Exemplary             | ``True``/``False``    | Is the dataset        |
|                       |                       | exemplary of the      |
|                       |                       | coordinate. Only one  |
|                       |                       | exemplary dataset is  |
|                       |                       | valid for each        |
|                       |                       | coordinate. Exemplary |
|                       |                       | datasets should have  |
|                       |                       | good coverage (both   |
|                       |                       | in keys space and     |
|                       |                       | feature space) and    |
|                       |                       | acceptable quality of |
|                       |                       | the data.             |
+-----------------------+-----------------------+-----------------------+
| Public                | ``True``/``False``    | Some datasets are     |
|                       |                       | public, and some are  |
|                       |                       | not, especially those |
|                       |                       | that come from        |
|                       |                       | collaborations with   |
|                       |                       | the pharma industry.  |
+-----------------------+-----------------------+-----------------------+

See the :mod:`chemicalchecker.database` for more information.

Dataset pre-processing
----------------------

Dataset pre-processing refers to everything that happens from
downloaded/calculated/user-defined data until Signature Type 0.
Pre-processing can be of very different complexity:

.. image:: img/preprocessing.png

Here is where most of the SB&NB research happens. For now, dataset
pre-processing is organized in a rather independent structure, i.e. each
dataset receives its pre-processing scripts
(see :mod:`chemicalchecker.core.preprocess`).