#!/bin/bash
#SBATCH --job-name=ML-BALANCED
#SBATCH --time=1-00:05:00
#SBATCH --mem=100GB
#SBATCH --partition=Main
#SBATCH --output=logs/ml-job-%j.out
#SBATCH --error=logs/ml-job-%j.err
#SBATCH --mail-user=pfunzowalter@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "---- Running ML job -----"

singularity exec /idia/software/containers/ASTRO-PY3.simg python EXP6_normalisation.py #ml-all-score-result-recall-vlbi.py #myslurmjob1.py #svm.py
# singularity exec /idia/software/containers/ASTRO-PY3.simg python svm.py