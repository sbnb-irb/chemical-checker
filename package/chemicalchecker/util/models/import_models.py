# Nico 29 Jan 2021 
# data (models to fit sign1 and sign2 are kept here)
# The following function allows a cc instance to copy the model files in a particular signature

import os, shutil
import h5py


def import_models(sign_object ,version='2020_01'):
    """
    Imports the models for predicting sign_objects
    i.e copies the models in the model_path of the reference signature object and
    create symbolics lincs to those model files in the full signature model_path
    """


    fileDir= os.path.abspath(os.path.dirname(__file__))

    signRef= sign_object.get_molset("reference")
    signFull= sign_object.get_molset("full")


    cctype= signRef.cctype
    dataset= signRef.dataset

    destination = signRef.model_path
    destinationLink = signFull.model_path


    data= os.path.join(fileDir,version,dataset,cctype)

    if not os.path.exists(data):
        print("Sorry, no model to import for this signature")
        return None

    for fichero in os.listdir(data):
        if fichero == "fit.ready":
            continue

        source= os.path.join(data,fichero)
        target= os.path.join(destination,fichero)
        symlink= os.path.join(destinationLink,fichero)


        if not os.path.exists(target):
            print("Importing", source, "to", target)
            shutil.copyfile(source, target)

        # Symlincs
        if not os.path.islink(symlink):
            print("Creating symlink", symlink, "from", target)
            os.symlink(target, symlink)

def import_sign0_features(sign_object, version='2020_01'):
    """
    Opens the stored sign0 h5 feature file and returns a set of features

    Arguments:
    sign_object : signature 0 object
    version (str): version of the cc_repo stored in package/chemicalchecker/util/models
    """

    cctype= sign_object.cctype
    dataset= sign_object.dataset

    if cctype != 'sign0' or not dataset.startswith('A'):
        print("Sorry, we only import features for sign0 A spaces")
        print("Requested", sign_object.cctype, sign_object.dataset)
        return None

    fileDir= os.path.abspath(os.path.dirname(__file__))
    featureFile= os.path.join(fileDir, version, dataset, 'sign0', 'features.h5')

    with h5py.File(featureFile, 'r') as fp:
        features = fp['features']
        return list(features)