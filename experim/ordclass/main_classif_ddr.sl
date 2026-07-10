#!/bin/bash

### Configure SLURM options
### Partition and walltime
### QoS	        Priority	MaxWall	    MaxNodesPU	MaxJobsPU	MaxSubmitPU	MaxTRES
### test        1000      0-00:10:00	2           2         2           -
### regular (D) 200       1-00:00:00	60	        180       -           -
### long        200       2-00:00:00	25	        40        -           -
### xlong       200       8-00:00:00	20          20        200         -
### serial      200       2-00:00:00	-           1000		  2000        cpu=1, gpu=0, node=1

#SBATCH --partition=general

#SBATCH --qos=regular
#SBATCH --time=1-00:00:00  # walltime (days-hours:minutes:seconds)

### Parallelization
#SBATCH --array=0-119

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G  # total memory

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-icelake

### Extra
#SBATCH --export=NONE

#SBATCH --job-name=OrdinClass
#SBATCH --output=../../results/DL/slurm-%A_%a.out

#SBATCH --mail-user=fegarcia@bcamath.org
#SBATCH --mail-type=ARRAY_TASKS,ALL

### Load software modules
module load Miniforge3
module load foss/2023a
module load cuDNN/8.7.0.84-CUDA-11.8.0

### Prepare environment
conda activate /scratch/fegarcia/conda-env/ordinal
cd ../..

### Run code
python3 -u -m experim.ordclass.main_classif_ddr --idx "$SLURM_ARRAY_TASK_ID"
