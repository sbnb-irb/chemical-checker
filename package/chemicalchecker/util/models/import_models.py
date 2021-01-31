# Nico 29 Jan 2021 
# data (models to fit sign1 and sign2 are kept here)
# The following function allows a cc instance to copy the model files in a particular signature

import os, shutil


def import_models(sign_object ,version='2020_01'):
    
    fileDir= os.path.abspath(os.path.dirname(__file__))


    cctype= sign_object.cctype
    dataset= sign_object.dataset
    molset=sign_object.molset
    destination=sign_object.model_path

    if molset != 'reference':
        print("Please use a reference signature (not full)")
        return None

    data= os.path.join(fileDir,version,dataset,cctype)
    if not os.path.exists(data):
        print("Sorry, no model to import for this signature")
        return None

    for fichero in os.listdir(data):
        source= os.path.join(data,fichero)
        target= os.path.join(destination,fichero)
        print("Importing", source, "to", target)
        shutil.copyfile(source, target)

