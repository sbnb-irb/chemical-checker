"""Config definition.

A Config is a JSON file specifying useful information throughout that are
necessary for the Chemical Checker.
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
                raise KeyError("CC_CONFIG environment variable not set.")
            except Exception as err:
                raise err
        self.__log.debug('Loading config from: %s' % json_file)
        with open(json_file) as fh:
            obj_dict = json.load(fh)
        eval_obj_dict = dict()
        for k, v in obj_dict.items():
            eval_obj_dict[k] = _Field(v)
        self.__dict__.update(eval_obj_dict)
        os.environ["CC_CONFIG"] = json_file


__all__ = [
    "Config"
]
