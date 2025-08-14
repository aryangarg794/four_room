#!/bin/sh
#SBATCH --partition=general # Request partition. Default is 'general' 
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=01:30:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=1   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=8192          # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=BEGIN,END,FAIL     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err # Set name of error log. %j is the Slurm jobId
#SBATCH --gres=gpu:1

export APPTAINER_HOME=/tudelft.net/staff-umbrella/ExploreGo Star/containers
export APPTAINER_NAME=image.sif
export PROJECT_HOME=/tudelft.net/staff-umbrella/ExploreGo Star/four_room
export RESULTS_DIR=/tudelft.net/staff-umbrella/ExploreGo Star/four_room/dqn_results

if [ ! -f $APPTAINER_HOME/$APPTAINER_NAME ]; then
    ls $APPTAINER_HOME/$APPTAINER_NAME
    exit 1
fi 
