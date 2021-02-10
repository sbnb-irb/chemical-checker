# Nico 10 Feb 2021
# Create symlinks for sign0 and 1 into a single destination directory

import os

version = "2020_01"
root="/aloy/web_checker/package_cc/"

destination = "/aloy/scratch/nsoler/CC_related/EXPORT_SIGN"

for molset in ('full',):
	for space in "ABCDE":
		for num in (1, 2, 3, 4, 5):
			for sign in ('sign0', 'sign1'):
				subsp= space+str(num)
				ds = subsp+'.001'
				signFile= os.path.join(root,version, molset,space, subsp, ds, sign, sign+'.h5')

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

