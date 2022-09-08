#!/bin/bash
# Sample batchscript to run a parallel python job on HPC using 10 CPU cores
#SBATCH --partition=bch-gpu                	# queue to be used
#SBATCH --time=30:00:00                         # Running time (in hours-minutes-seconds)
#SBATCH --job-name=main                 	# Job name
#SBATCH --mail-type=BEGIN,END,FAIL              # send and email when the job begins, ends or fails
#SBATCH --mail-user=andy.tsai@childrens.harvard.edu          # Email address to send the job status
#SBATCH --output=main-output_%j.txt                  # Name of the output file
#SBATCH --nodes=1                               # Number of compute nodes
#SBATCH --ntasks=10                             # Number of cpu cores on one node
#SBATCH --gres=gpu:Tesla_T:3			# Number of gpu devices on 1 gpu node
#SBATCH --mem=20GB

source /programs/biogrids.shrc
python.tensorflow main.py
