import faiss  # will be required by jobs
from chemicalchecker.util.pipeline import Pipeline, CCFit

data_dir = './tests/data'
pipeline_dir = './dpipe'
data_file = os.path.join( data_dir, 'E1_preprocess.h5')
cc_root = os.path.join(pipeline_dir, 'cc')

pp = Pipeline(pipeline_path=pipeline_dir, config=Config())
print(os.path.isdir(pp.readydir))

# SIGN 0
s0_fit_kwargs = {
    "E1.001": {
        'key_type': 'inchikey',
        'data_file': data_file,
        'do_triplets': False,
        'validations': False
    }
}
s0_task = CCFit(cc_root, 'sign0', 'full',
                datasets=['E1.001'], fit_kwargs=s0_fit_kwargs)
pp.add_task(s0_task)
pp.run()

sign0_full_file = os.path.join(
    cc_root, 'full/E/E1/E1.001/sign0/sign0.h5')
print(os.path.isfile(sign0_full_file))
sign0_ref_file = os.path.join(
    cc_root, 'reference/E/E1/E1.001/sign0/sign0.h5')
print(os.path.isfile(sign0_ref_file))

# SIGN 1
s1_fit_kwargs = {
    "E1.001": {
        'metric_learning': False,
        'validations': False
    }
}
s1_task = CCFit(cc_root, 'sign1', 'full',
                datasets=['E1.001'], fit_kwargs=s1_fit_kwargs)
pp.add_task(s1_task)
pp.run()

sign1_full_file = os.path.join(
    cc_root, 'full/E/E1/E1.001/sign1/sign1.h5')
print(os.path.isfile(sign1_full_file))
sign1_ref_file = os.path.join(
    cc_root, 'reference/E/E1/E1.001/sign1/sign1.h5')
print(os.path.isfile(sign1_ref_file))

# NEIG 1
s1_neig_task = CCFit(cc_root, 'neig1', 'reference',
                     datasets=['E1.001'])
pp.add_task(s1_neig_task)
pp.run()
neig1_ref_file = os.path.join(
    cc_root, 'reference/E/E1/E1.001/neig1/neig.h5')
print(os.path.isfile(neig1_ref_file))

s2_fit_kwargs = {
    "E1.001": {
        'validations': False,
        'oos_predictor': False
    }
}
s2_task = CCFit(cc_root, 'sign2', 'reference',
                datasets=['E1.001'], fit_kwargs=s2_fit_kwargs)
pp.add_task(s2_task)
pp.run()

# SIGN 2
sign2_ref_file = os.path.join(
    cc_root, 'reference/E/E1/E1.001/sign2/sign2.h5')
print(os.path.isfile(sign2_ref_file))
