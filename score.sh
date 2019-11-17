#!/bin/bash
#$ -m abes -M joao.monteiro@crim.ca
#$ -cwd
#$ -o $JOB_NAME.$JOB_ID.out
#$ -e $JOB_NAME.$JOB_ID.err
#$ -v PATH
#$ -l arch=lx*
#$ -q q.i9gpu
#$ -l mem_free=32000M

python score.py \
--path-to-data /lu/bf_scratch/patx/alamja/asvspoof2019challenge/fisher_vectors/comblfhfc_both_32/ivectors_eval/ivector.scp \
--cp-path /misc/data18/patx/monteijo/Nobackup/cp/spoofing/fv/858267309.pt \
--trials-path /lu/bf_scratch/patx/alamja/blind_ssd2019/trials/pa_eval.trl.txt \
--out-path ./scores/eval.out
