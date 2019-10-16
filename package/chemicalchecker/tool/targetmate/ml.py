"""TargetMate ML classes"""

from chemicalchecker.util import logged
from .base import FingerprintModel, EnsembleModel, StackedModel


@logged
class FingerprintClassifier(FingerprintModel):
    """ """
    
    def __init__(self, **kwargs):
        FingeprintModel.__init__(self, is_classifier=True, **kwargs)    


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

