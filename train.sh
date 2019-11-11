#!/bin/bash
#$ -m abes -M joao.monteiro@crim.ca
#$ -cwd
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err
#$ -v PATH
#$ -l arch=lx*
#$ -q q.gpu
#$ -l mem_free=24000M
#$ -l cuda_capability=6.1
#$ -l h_core=4

python train.py \
--train-hdf-path /misc/scratch07/patx/monteijo/spoofing/fv/ \
--valid-hdf-path /misc/scratch07/patx/monteijo/spoofing/fv/ \
--save-every 20 \
--epochs 500 \
--checkpoint-path /misc/data18/patx/monteijo/Nobackup/cp/spoofing/fv \
--workers 4 \
--batch-size 8 \
--seed 242055 \
--lr 1.5 \
--momentum 0.9 \
--l2 1e-3 \
--n-cycles 1 \
--valid-n-cycles 10
