"""Identity method, i.e. nothing changes"""

from .base import BaseTransform

class Identity(BaseTransform):

    def __init__(self, sign1):
        BaseTransform.__init__(self, sign1, "identity")

    def fit(self, **kwargs):
        self.save()

    def predict(self, **kwargs):
        pass