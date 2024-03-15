#!/bin/bash
#
# CREATED USING THE BIOHPC PORTAL on Fri Mar 15 2024 03:07:14 GMT-0500 (Central Daylight Time)
#
# This file is batch script used to run commands on the BioHPC cluster.
# The script is submitted to the cluster using the SLURM `sbatch` command.
# Lines starting with # are comments, and will not be run.
# Lines starting with #SBATCH specify options for the scheduler.
# Lines that do not start with # or #SBATCH are commands that will run.

# Name for the job that will be visible in the job queue and accounting tools.
#SBATCH --job-name egg_generic_oral_ceograph_class_0_run_1

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPUv100s       # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

# Memory (RAM) requirement/limit in MB.
#SBATCH --mem 380928      # Memory Requirement (MB)

# Time limit for the job in the format Days-H:M:S
# A job that reaches its time limit will be cancelled.
# Specify an accurate time limit for efficient scheduling so your job runs promptly.
#SBATCH -t 0-2:0:0

# The standard output and errors from commands will be written to these files.
# %j in the filename will be replace with the job number when it is submitted.
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Send an email when the job status changes, to the specfied address.
#SBATCH --mail-type ALL
#SBATCH --mail-user artit.taychameekiatchai@utsouthwestern.edu

unset CUDA_VISIBLE_DEVICES
module load python/3.8.x-anaconda
module load cuda121
module load cudnn

# COMMAND GROUP 1
cd
cd /work/DPDS/s24833/Dissertation/gnn-egg/

source /cm/shared/apps/python/3.8.x-anaconda/etc/profile.d/conda.sh
conda activate egg-env-38

python scripts/train_egg_generic_oral_ceograph.py --path_to_json_config experiments/egg_generic/class_0_run_1/config.json

# END OF SCRIPT