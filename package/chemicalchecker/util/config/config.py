"""Chemical Checker config.

The config is defined in a JSON file (i.e. ``cc_config.json``) and it holds
information that are necessary throughout the Chemical Checker.

The path to this file can be passed to the :class:`Config`
constructor or it can be set in the environment variable ``CC_CONFIG``. In
this case the specified config will be used by default.

When instantiated the variables will be attributes of the config instance::

    from chemicalchecker.util import Config
    cfg = Config()
    cfg.PATH.CC_TMP
    >>> /tmp/

An example of this file will contain::

    {
        "PATH": {
            "CC_ROOT": "'/aloy/web_checker/current'",
            "CC_DATA": "'/aloy/scratch/oguitart/download/'",
            "CC_TMP": "'/tmp/'",
            "CC_REPO": "'/opt/chemical_checker'",
            "SINGULARITY_IMAGE": "'/aloy/web_checker/images/cc.simg'",
            "validation_path": "'/aloy/web_checker/package_cc/validation_sets/'"
        },
        "DB": {
            "dialect": "'postgresql'",
            "port":  "'5432'",
            "host": "'10.5.4.240'",
            "user": "'cc_user'",
            "password": "'checker'",
            "database": "'cc_package'",
            "calcdata_dbname": "'cc_calcdata'",
            "uniprot_db_version" : "'2019_01'"
        },
        "HPC": {
            "system" : "'sge'",
            "host": "'pac-one-head'",
            "queue": "'all.q'",
            "username": "'cc_user'"
        }
    }

As you can see json fields are organized sections. The most
important is the ``PATH`` which contains the following information:

    * ``PATH.CC_ROOT`` default root directory of a CC instance (see
       :mod:`~chemicalchecker.core.chemcheck`).
    * ``PATH.CC_DATA`` directory for data download (see
       :mod:`~chemicalchecker.util.download`).
    * ``PATH.CC_TMP`` directory for temporary data.
    * ``PATH.CC_REPO`` directory of the default CC package.
    * ``PATH.SINGULARITY_IMAGE`` path to the singularity image used to
      submit HPC jobs (see :mod:`~chemicalchecker.util.hpc`)
    * ``PATH.validation_path`` directory holding validation set for ROC
      AUC calculation.

The section ``DB`` holds parameters needed to connect to local DBS (used
in :mod:`chemicalchecker.database`). The ``HPC`` section contain
parameters used in HPC job submission (used in :mod:`chemicalchecker.util.HPC`).
"""
import os
import json

from chemicalchecker.util import logged


class _Field():
    """Config Field placeholder."""

    def __init__(self, field_kv):
        """Initialize updating __dict__ and evaluating values."""
        tmp = dict()
        for k, v in field_kv.items():
            if type(v) == dict:
                tmp[k] = _Field(v)
            else:
                tmp[k] = eval(v)
        self.__dict__.update(tmp)

    def items(self):
        return self.__dict__.items()

    def asdict(self):
        return self.__dict__

    def __getitem__(self, key):

        return self.__dict__[key]


@logged
class Config():
    """The persistent container of a config file.

    The Config provide access to all sort of useful parameters.
    """

    def __init__(self, json_file=None):
        """A Config is loaded from a JSON file."""
        if not json_file:
            try:
                json_file = os.environ["CC_CONFIG"]
            except KeyError as err:
                self.__log.debug("CC_CONFIG environment variable not set. " +
                                 "Using default config file.")
                json_file = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), 'cc_config.json')

            except Exception as err:
                raise err
                
        self.__log.debug('Loading config from: %s' % json_file)
        with open(json_file) as fh:
            obj_dict = json.load(fh)

        eval_obj_dict = dict()
        for k, v in obj_dict.items():
            if type(v) == dict:
                eval_obj_dict[k] = _Field(v)
            else:
                eval_obj_dict[k] = eval(v)
        self.__dict__.update(eval_obj_dict)
        #os.environ["CC_CONFIG"] = json_file

    def keys(self):
        return self.__dict__.keys()


__all__ = [
    "Config"
]
