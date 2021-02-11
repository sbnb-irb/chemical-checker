# Nico 11 Feb 2021
# export sign1 models into the cc code repo directory:

import os, shutil, glob

from get_repo_version import cc_repo_version

def export_models(destination, cctype='sign1', cc_repo=None):


    if cc_repo is None:
        cc_repo = cc_repo_version()

        if cc_repo is None:
            print("ERROR, cannot guess the latest cc repository path")
            print("Please provide it as an argument")
            print("ex: cc_repo='/aloy/web_checker/package_cc/2020_02'")
            return
        else:
            print("Working with cc_repo:",cc_repo)


    for space in ['A'+str(n) for n in range(1,6)]:
        ds= space+str(num)+'.001'
        dirModels= os.path.join(cc_repo,'reference', 'A', space, ds, cctype,'models')
        dirtmp= os.path.join(destination,ds,cctype)

        if not os.path.exists(dirtmp):
            try:
                os.makedirs(dirtmp)
            except Exception as e:
                print("ERROR while attempting to create destination folder", dirtmp)
                print(e)
                continue
            else:
                print("Created directory", destination)

        for fichero in glob.glob(dirModels):
            if fichero != 'fit.ready':
                source= os.path.join(dirModels,fichero)
                target= os.path.join(dirtmp,fichero)
                try:
                    shutil.copyfile(source, target)
                except Exception as e:
                    print(e)
                else:
                    print("Copied:", source, 'to' , target)

if __name__=='__main__':

    destination = "/home/nsoler/CODE/chemical_checker/package/chemicalchecker/util/models/2020_02"
    export_models(destination=destination)
