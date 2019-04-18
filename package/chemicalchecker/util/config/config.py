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


__all__ = [
    "Config"
]
