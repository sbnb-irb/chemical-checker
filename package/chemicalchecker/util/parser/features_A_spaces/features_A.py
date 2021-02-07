import h5py, os


def fetch_features_A():
    """
    Returns a dictionary {space, features} for all chemistry space (A)
    """

    # The h5 data files are stored in the same directory
    data={'A'+str(n): 'features_A'+str(n)+'_001.h5' for n in (1, 2, 3, 4, 5)}

    fileDir= os.path.abspath(os.path.dirname(__file__))

    out= dict()
    for space, h5file in data.items():
        fichero= os.path.join(fileDir,h5file)
        print("Loading",fichero,"for space",space)
        
        with h5py.File(fichero,'r') as fp:
            out[space] = fp['features'][:]

    return out
