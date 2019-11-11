#!/bin/bash
#$ -m abes -M joao.monteiro@crim.ca
#$ -cwd
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err
#$ -v PATH
#$ -l arch=lx*
#$ -q q.i9 -q q.all
#$ -l mem_free=32000M

python data_prep.py --path-to-data   /lu/bf_scratch/patx/alamja/asvspoof2019challenge/fisher_vectors/comblfhfc_both_32/ivectors_train_bonafide/ivector.scp --out-path /misc/scratch07/patx/monteijo/spoofing/fv/ --out-name clean.hdf --n-val-speakers 100 \
&& python data_prep.py --path-to-data   /lu/bf_scratch/patx/alamja/asvspoof2019challenge/fisher_vectors/comblfhfc_both_32/ivectors_train_spoof/ivector.scp --out-path /misc/scratch07/patx/monteijo/spoofing/fv/ --out-name attack.hdf --n-val-speakers 1000
