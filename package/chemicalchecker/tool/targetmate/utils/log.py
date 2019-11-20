import logging

def set_logging(log="INFO"):
    if log == "ERROR":
        logging.getLogger().setLevel(logging.ERROR)
    if log == "WARNING":
        logging.getLogger().setLevel(logging.WARNING)
    if log == "INFO":
        logging.getLogger().setLevel(logging.INFO)
    if log == "DEBUG":
    	logging.getLogger().setLevel(logging.DEBUG)
