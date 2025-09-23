#!/bin/bash

#SBATCH --nodelist=gpu-[001-002,004-005]
#SBATCH --nodes=1
#SBATCH --job-name=ML-Reviews
#SBATCH --time=4-00:05:00
#SBATCH --mem=100GB
#SBATCH --partition=GPU
#SBATCH --output=logs/ml-job-%j.out
#SBATCH --error=logs/ml-job-%j.err
#SBATCH --mail-user=pfunzowalter@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

echo "---- Running ML job -----"

singularity exec /idia/software/containers/ASTRO-PY3.simg python EXP6_reviews_recall.py
# singularity exec /idia/software/containers/ASTRO-PY3.simg python svm.py
