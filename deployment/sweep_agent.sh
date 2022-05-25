#!/bin/bash
#
#SBATCH --job-name=20ng_distilbert_sweep
#
#SBATCH --mem-per-gpu=5g
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.5gb
#SBATCH --gpu-bind=single:1
#SBATCH --nodes=1
#SBATCH --output=/dev/null
#SBATCH -e slurm-%j.err

srun \
--export=ALL \
--container-image=./20ng.sqsh \
--container-mounts=/mnt/ceph:/mnt/ceph \
--container-name=sweep_container-$1 \
bash -c " cd ~ && PYTHONPATH=. WANDB_ENTITY=$WANDB_ENTITY WANDB_PROJECT=$WANDB_PROJECT WANDB_CONSOLE=off python3 $SWEEP_AGENT --sweep_id=$SWEEPID "