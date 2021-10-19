# TargetMate

TargetMate contains functionalities to do **supervised machine learning** with the Chemical Checker.

| | Ensemble | Stacked |
| Classifcation |
| Regression | ::tick:: |

## Steps followed by TargetMate

### Fitting time

1. Identify a good pipeline through the use of CV grid search or TPOT.
   - Full data.
2. Cross-conformal prediction
3. A stacked classifier based
4. An ensemble-based classifier based on Chemical Checker signatures.
    * A base classifier is specified, and predictions are made for each dataset individually.
    * An ensemble-based prediction is then given based on individual
    predictions. Proficient base classifiers contribute more to the prediction.
    * Conformal prediction is used to evaluate the applicability domain and confidence of the prediction.

For information on how to do conformal prediction, please see: https://arxiv.org/pdf/1908.03569.pdf .

Intuitively, the non-conformity measure quantifies how different a given instance is
from those already seen during the training phase, which is quantified using a non-conformity
score. Therefore, any metric quantifying the applicability domain of a model, e.g., the
distance of a new instance to the training set, can be used as non-conformity measure

### Term Definition
#### Validity
Conformal Predictors are always valid provided that the randomness or
exchangeability principles hold. In the case of regression, a Conformal
Predictor is valid if the confidence level matches the fraction of instances
whose true value lies within the predicted confidence region. For instance,
at a confidence level of 80%, the confidence intervals would contain the
true value in at least 80% of the cases. In classification tasks, Conformal
Predictors are valid in that the set of predicted classes for new instances
will contain the true label in at least 80% of the cases.
#### Efficiency
In regression, the efficiency of a CP refers to the average size of the
predicted confidence intervals. The tighter the intervals the more efficient a
conformal predictor is. In the case of classification, efficiency refers to the
fraction of single-class predictions that are correct.
#### Confidence level (CL)
The confidence level is defined by the modeler and refers to the minimum
fraction of predictions whose true value will lie within the predicted
confidence region, in the case of regression, and the fraction of instances
whose true class will be among the set of predicted classes.
#### Error rate
The error rate refers to the fraction of instances whose true value lies
outside the predicted confidence regions. If a CP is well-calibrated, the
error rate should not be larger than 1-CL (see also Figure 1).
#### Nonconformity measure
Function used to evaluate the relatedness or conformity of new instances
to those used for model training.

## Mondrian conformal prediction (MCP)

In MCP each class (e.g., active and inactive) is treated separately, and the confidence in the
assignment of a given instance to the classes considered is evaluated independently. That is, a
list of non-conformity scores is generated for each class using the predictions for the calibration
set (Figure 4). Thus, in a binary classification setting a compound might be classified as “active”,
“inactive”, both active and inactive (class “both”), or not assigned to either of them (class “null”
or “empty”).

1. Choosing a non-conformity measure to evaluate the non-conformity between the training
and the test instances;
2. Training the machine learning model of choice, and evaluate the non-conformity values
for the training examples;
3. Applying the trained model to the test or external set instances;
4. For each test set instance, evaluating its non-conformity with respect to the training data
using the same non-conformity measure used in step 1: the higher the conformity of the
new instance, the higher the reliability of the prediction;
5. Identifying reliable predictions given the user-defined significance and confidence levels;
and
6. Evaluating the validity and efficiency of the generated Conformal Predictor.

## Create a hierarchy of ChEMBL assays

First, get the ChEMBL targets of your interest. Below, we fetch targets related to bacteria, viruses and fungi.

```python3
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

```python3
chembl = Chembl("chembl_infectious_assays")
chembl.write_folder_hierarchy(target_chembl_ids = target_ids)
```
### Alternative: DrugBank data
Activity data can also be collected from DrugBank, this allows the use of simple active/inactive class (i.e. drugs which interact with a certain target) or multiclass labels (e.g. agonsit, antagonist).

```python3
db = DrugBank("drugbank_infectious_assays", uniprot_acs= dream_targets)
db.parse_activities()
```

## Compute a small molecule universe

It is necessary to have a _universe_ of molecules as, oftentimes, actives are available and inactives are not for a particular target. The universe is just the pool of molecules where putative inactives are sampled from. By default, TargetMate uses ChEMBL as the universe.

```python3
from chemicalchecker.tool.targetmate.universes import Universe

universe = Universe(model_path = "path/to/universe", molrepo = "drugbank")
universe.fit()
universe.save()
```

## Set up a model
_TargetMate_ can be customized as desired, though some parameters are required for the default model.

* **data**: Root of folder where data is stored, this may also be an explicit path if creating a model for a single protein
* **models_path**: root to store models, these will be stored or deleted after validation according to selected options
* **overwrite**: boolean whether or to overwrite previously stored models with same name
* **TargetMate**: Instance of type of model to be created, at present only Classifier models are available
* **use_cc**: Whether or not to use ChemicalChecker signatures (ADD CITATION)


## Run model
An assortment of model options are available (for more details, see ![Types of models]), importantly both the algorithm and the model configuration must be set.

### Basic Set Up
For these we use `MultiValidation` which allows the creation of models for one or more protein, based on the same type of model but training on each protein's dataset.

#### Validate Model
```python3
mv = MultiValidation(data, models_path, overwrite)

mv.run(TargetMate, use_cc, use_inchikey)
```

#### Validate and store model

```python3
mv = MultiValidation(data, models_path, overwrite, wipe=True, keep_only_validations=False)

mv.run(TargetMate, use_cc, use_inchikey,is_tmp=False)
```

#### Use all data for training

```python3
mv = MultiValidation(data, models_path, overwrite)

mv.run(TargetMate, use_cc, use_inchikey, only_train=True)
```

## Validate model

If you have chosen to carry out validation when running a model, two files will be created for each protein model (`validation.txt` and `validation.pkl`). `validation.txt` contains the chosen metric score for easy visualisation (AUROC by default). `validation.pkl` contains more detailed information regarding the validation of the models, such as y_pred and several performace metrics.

## Use model

If you have chosen to store fitted models these can be used to predict compound's binding probabilities. This is carried out using the `Prediction` class.

```python3
pred = Prediction(model_root, data, output_directory)

pred.run()
```
