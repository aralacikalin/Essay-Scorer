#!/bin/bash

# The name of the job is test_job
#SBATCH -J train_mt_exp_classifier

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=log.out


# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=1

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:a100-80g:1



# The maximum walltime of the job is 5 minutes
#SBATCH -t 24:00:00

#SBATCH --mem=60G

##SBATCH --mail-type=ALL
##SBATCH --mail-user=aral.ackaln@ut.ee

# Commands to execute go below

# Load Python
module load any/python/3.8.3-conda

# Activate your environment
#conda activate mtcourse

# Display fairseq's help message
COMET_API_KEY="xLyDNwJVqiVwJcwFlcaSC2Urm" COMET_PROJECT_NAME="mt-project" python -u train_classifier.py
