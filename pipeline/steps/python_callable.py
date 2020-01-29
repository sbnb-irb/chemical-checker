import tempfile
import os
import shutil

from chemicalchecker.util import logged
from chemicalchecker.util import BaseTask


@logged
class PythonCallable(BaseTask):

    def __init__(self, config=None, name='python', **params):

        BaseTask.__init__(self, config, name, **params)

        self.python_callable = params.get('python_callable', None)
        self.op_args = params.get('op_args', [])
        self.op_kwargs = params.get('op_kwargs', {})

        if self.python_callable is None:
            raise Exception("PythonCallable task requires a callable object")

    def run(self):

        return_value = self.python_callable(*self.op_args, **self.op_kwargs)

        self.__log.info("Done. Returned value was: %s", return_value)
        self.mark_ready()
