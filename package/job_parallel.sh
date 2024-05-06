#!/bin/bash -i
#
#

# Options for qsub
#$ -S /bin/bash
#$ -r yes
#$ -j yes
#$ -q all.q@pac-one301,all.q@pac-one401
#$ -N CC_CHEMPROP_PARALLEL
#$ -wd /aloy/home/ymartins/Documents/cc_update/chemical_checker/package
#$ -t 30
#$ -pe make 30
#$ -l mem_free=30G,h_vmem=40.2G
# End of qsub options

# Loads default environment configuration
if [[ -f $HOME/.bashrc ]]
then
  source $HOME/.bashrc
fi


OMP_NUM_THREADS=30 OPENBLAS_NUM_THREADS=30 MKL_NUM_THREADS=30 VECLIB_MAXIMUM_THREADS=30 NUMEXPR_NUM_THREADS=30 NUMEXPR_MAX_THREADS=30 SINGULARITYENV_PYTHONPATH=/aloy/home/ymartins/Documents/cc_update/chemical_checker/package SINGULARITYENV_CC_CONFIG=/aloy/home/ymartins/Documents/cc_update/chemical_checker/pipelines/configs/cc_package.json singularity exec /aloy/home/ymartins/Documents/cc_update/cc_image/cc.simg python /aloy/home/ymartins/Documents/cc_update/chemical_checker/package/test_e3fp_parallel.py
