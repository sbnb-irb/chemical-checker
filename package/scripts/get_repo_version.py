# Nico 11 Feb 2021
# Put here the current CC signature repo version that can be then fetched by scripts

import os

CURRENT = "/aloy/web_checker/current"

def cc_repo_version():
	try:
		return os.path.abspath(os.path.realpath(CURRENT))
	except Exception as e:
		print("WARNING: the cc signature repo version cannot be retrieved from", CURRENT)
		print(e)
		return None
