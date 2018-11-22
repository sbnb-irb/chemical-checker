"""Recipe definition.

A Recipe is a JSON file specifying network elements to include, classes for
Network, NetworkSource, Cleaner, MetaPath, RandomWalker and Embedding, and
their respective parameters.
"""
import os
import json


class _Field():
    """Recipe Field placeholder."""

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


class Config():
    """The persistent container that determine the embedding.

    Nodes and Edges type, attributes, and weights to consider are listed here.
    The Config provide access to all parameters of Cleaner, MetaPath,
    RandomWalker, and Embedding as well as all the specific classes to use
    for each component.
    """

    def __init__(self, json_file=None):
        """A Config is loaded from a JSON file."""
        #self.__log.debug('Loading recipe from: %s' % json_file)
        if not json_file:
            try:
                json_file = os.environ["CC_CONFIG"]
            except KeyError as err:
                raise KeyError("CC_CONFIG enviroment variable not set.")
            except Exception as err:
                raise err

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
