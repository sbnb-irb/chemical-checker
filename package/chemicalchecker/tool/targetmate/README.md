# TargetMate classifier

An ensemble-based classifier based on Chemical Checker signatures.

* A base classifier is specified, and predictions are made for each dataset individually.
* An ensemble-based prediction is then given based on individual
predictions. Proficient base classifiers contribute more to the prediction.
 * Together with a measure of confidence for each prediction.
 * 
In the predictions, known data is provided as 1/0 predictions. The rest of
probabilities are clipped between 0.001 and 0.999.
In order to make results more interpretable, in the applicability domain we
use chemical similarity for now.
The basis for the applicability domain application can be found:
https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0182-y
Obviously, CC signature similarities could be used in the future.
The classifier is greatly inspired by PidginV3:
https://pidginv3.readthedocs.io/en/latest/usage/index.html
