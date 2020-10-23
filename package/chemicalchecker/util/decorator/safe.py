"""Decorator that nicely handle exceptions in functions."""
import traceback
from functools import wraps

from chemicalchecker.util import logged


@logged
def safe_return(ex_return):
    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as ex:
                safe_return._log.warning('* Exception: %s' % ex)
                safe_return._log.warning('* safe_return: %s' % ex_return)
                tb_lines = [l.rstrip('\n') for l in traceback.format_exception(
                    ex.__class__, ex, ex.__traceback__)]
                for line in tb_lines:
                    safe_return._log.warning('* %s' % line)
                return ex_return
        return wrapper
    return inner_function
