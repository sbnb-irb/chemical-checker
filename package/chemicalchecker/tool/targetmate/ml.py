"""TargetMate ML classes"""

@logged
def FingerprintClassifier(TargetMateClassifier, Fingerprinter):
    """ """
    
    def __init__(self, **kwargs):
        FingeprintModel.__init__(self, is_classifier=True, **kwargs)    


@logged
def TargetMateStackedClassifier(StackedModel, Signaturizer):
    """Stacked predictions"""


@logged
def TargetMateEnsembleClassifier(EnsembleModel, Signaturizer):
    """Ensemble predictions targetmate"""