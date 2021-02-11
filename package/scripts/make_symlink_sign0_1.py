# Nico 10 Feb 2021
# Create symlinks for sign0 and 1 into a single destination directory

import os, sys
from get_repo_version import cc_repo_version

def make symlinks(destination = "/aloy/scratch/nsoler/CC_related/EXPORT_SIGN", cc_repo=None):
	"""
	Creates symlinks for all signatures in a single folder

	"""

    if cc_repo is None:
    	cc_repo = cc_repo_version()

	    if cc_repo is None:
	        print("ERROR, cannot guess the latest cc repository path")
	        print("Please provide it as an argument")
	        print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
	        return
	    else:
	    	print("Working with cc_repo:",cc_repo)

	if not os.path.exists(destination):
		try:
			os.makedirs(destination)
		except Exception as e:
			print("ERROR while attempting to create destination folder", destination)
			print(e)
		else:
			print("Created directory", destination)

	for molset in ('full', 'reference'):
		for space in "ABCDE":
			for num in (1, 2, 3, 4, 5):
				for sign in ('sign0', 'sign1','sign2','sign3'):
					subsp= space+str(num)
					ds = subsp+'.001'
					signFile= os.path.join(cc_repo, molset,space, subsp, ds, sign, sign+'_BACKUP.h5')

					if os.path.exists(signFile):
						# Make a symlink into the destination
						symlink = os.path.join(destination, sign+'_'+subsp+'_'+molset+'.h5')
						try:
							os.symlink(signFile, symlink)
						except Exception as e:
							print("Error for creating", symlink)
							print(e)
						else: 
							print("Created symlink:",symlink)

					else:
						print("File not found: ",signFile)

if __name__== '__main__':

	destination = ""