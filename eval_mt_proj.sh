#!/bin/bash

# The name of the job is test_job
#SBATCH -J eval_mt_distil_regression_smartTrunc

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=nlp-%x.%j.out


# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=1

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
## --gres=gpu:a100-40g:1
#SBATCH --gres=gpu:tesla:1



# The maximum walltime of the job is 5 minutes
#SBATCH -t 02:00:00

#SBATCH --mem=60G


# Commands to execute go below

# Load Python
module load any/python/3.8.3-conda

# Activate your environment
conda activate mtcourse

# Display fairseq's help message
/gpfs/space/home/aral/.conda/envs/mtcourse/bin/python /gpfs/space/home/aral/mtProject/evaluate_regression.py