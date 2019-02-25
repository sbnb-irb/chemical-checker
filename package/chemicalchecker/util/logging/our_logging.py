
import os as _os
import logging.config as _logging

_log_conf = _os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), 'logging_conf.ini')
_logging.fileConfig(_log_conf)
