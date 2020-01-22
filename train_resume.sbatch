#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=DeepVO_train
#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=32G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=6-12:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=claudio.raisch@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

#echo "extract data to /scratch/$SLURM_JOB_ID:"
#tar -xvzf /home/raisch/Datasets/KITTI/DeepVO-KITTI.tar.gz -C /scratch/$SLURM_JOB_ID
# TODO figure out how to extract only a subset of the data in tar archive
echo "copy data to /scratch/$SLURM_JOB_ID:"
mkdir /scratch/$SLURM_JOB_ID/images /scratch/$SLURM_JOB_ID/poses_gt
ids=( "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" )
for id in "${ids[@]}"
do
     echo "\tcopy sequence $id..."
     rsync --info=progress2 -r /home/raisch/Datasets/KITTI/images/$id /scratch/$SLURM_JOB_ID/images
     rsync --info=progress2 /home/raisch/Datasets/KITTI/poses_gt/$id.* /scratch/$SLURM_JOB_ID/poses_gt
done
ls -l /scratch/$SLURM_JOB_ID
ls -l /scratch/$SLURM_JOB_ID/images
ls -l /scratch/$SLURM_JOB_ID/poses_gt
du -sh /scratch/$SLURM_JOB_ID

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

IMG_FILE=/common/singularityImages/TCML-Cuda10_0Tensorflow2_0_0.simg
HOME=/home/raisch/DeepVO-pytorch

echo "executing training..."
singularity exec --nv $IMG_FILE python3 $HOME/main.py --remote_dir=/scratch/$SLURM_JOB_ID --home_dir=$HOME --resume
echo " done!"

