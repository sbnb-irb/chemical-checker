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
parameters used in HPC job submission
(used in :mod:`chemicalchecker.util.HPC`).
"""
from .config import Config
