"""Logging configuration.

In the CC package we use the
`autologging package <http://ninthtest.info/python-autologging/>`_
which offers the handy ``@logged`` decorator for  classes or function.

.. note::
   The chain of imports is important, so relevant imports for this module are
   done at the level of :mod:`~chemicalchecker.util`.

The loggers, default verbosity and output style is defined in the file
``logging_conf.ini``
"""
