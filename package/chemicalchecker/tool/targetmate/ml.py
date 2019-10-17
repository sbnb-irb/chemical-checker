"""TargetMate ML classes"""

from chemicalchecker.util import logged
from .base import FingerprintedModel, EnsembleModel, StackedModel


@logged
class FingerprintClassifier(FingerprintedModel):
    """ """
    
    def __init__(self, **kwargs):
        FingeprintedModel.__init__(self, is_classifier=True, **kwargs)    


@logged
class TargetMateStackedClassifier(StackedModel):
    """Stacked predictions"""
    
    def __init__(self, **kwargs):
        StackedModel.__init__(self, is_classifier=True, **kwargs)

@logged
class TargetMateEnsembleClassifier(EnsembleModel):
    """Ensemble predictions targetmate"""

    def __init__(self, **kwargs):
        EnsembleModel.__init__(self, is_classifier=True, **kwargs)

