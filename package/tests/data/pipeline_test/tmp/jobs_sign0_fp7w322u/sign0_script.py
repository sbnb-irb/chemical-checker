
import sys
import os
import pickle
import logging

sys.path.insert(0, '/aloy/home/ymartins/Documents/cc_update/chemical_checker/package/' )
import chemicalchecker
from chemicalchecker import ChemicalChecker, Config

logging.log(logging.DEBUG, 'chemicalchecker: {}'.format(
    chemicalchecker.__path__))
logging.log(logging.DEBUG, 'CWD: {}'.format(os.getcwd()))
config = Config()
task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>
inputs = pickle.load(open(filename, 'rb'))  # load pickled data
sign_args = inputs[task_id][0][0]
sign_kwargs = inputs[task_id][0][1]
fit_args = inputs[task_id][0][2]
fit_kwargs = inputs[task_id][0][3]
cc = ChemicalChecker('/mnt/085dd464-7946-4395-acfd-e22026d52e9d/home/yasmmin/backup/irbBCN_job/chemicalChecker/chemical_checker/package/tests/data/pipeline_test/cc')
sign = cc.get_signature(*sign_args, **sign_kwargs)
sign.fit(*fit_args, **fit_kwargs)
print('JOB DONE')
