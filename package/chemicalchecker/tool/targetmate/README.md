# TargetMate classifier

An ensemble-based classifier based on Chemical Checker signatures.

* A base classifier is specified, and predictions are made for each dataset individually.
* An ensemble-based prediction is then given based on individual
predictions. Proficient base classifiers contribute more to the prediction.
* Conformal prediction is used to evaluate the applicability domain and confidence of the prediction.

### Create a hierarchy of ChEMBL assays

First, get the ChEMBL targets of your interest. Below, we fetch targets related to bacteria, viruses and fungi.

```python
from chemicalchecker.tool.targetmate.datasets import Chembl

query = """
SELECT
    td.chembl_id
FROM
    target_dictionary td,
    organism_class oc
WHERE
    td.chembl_id IS NOT NULL
    AND oc.tax_id = td.tax_id
    AND oc.l1 IN ('Bacteria', 'Viruses', 'Fungi')
"""
target_ids = sorted(set([r[0] for r in psql.qstring(query, "chembl_25")]))
```

Now we create a hierarchy of files representing actives (1) and inactives (-1) in ChEMBL, according to the pChEMBL score.

```python
chembl = Chembl("chembl_infectious_assays")
chembl.write_folder_hierarchy(target_chembl_ids = target_ids)
```
#### Compute a small molecule universe

It is necessary to have a _universe_ of molecules as, oftentimes, actives are available and inactives are not for a particular target. The universe is just the pool of molecules where putative inactives are sampled from. By default, TargetMate uses ChEMBL as the universe.

```python
from chemicalchecker.tool.targetmate import Universe

universe = Universe(model_path = "path/to/universe", molrepo = "drugbank")
universe.fit()
universe.save()
```

### Train a model

```python



```


