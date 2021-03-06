#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=DVO-train-resume
#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=32G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=7-00:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

# NOTE fill in correct path
#SBATCH --error=/home/raisch/DeepVO-pytorch/experiments/sample_name/job.%J.resume.err
# write the error output to job.*jobID*.err

# NOTE fill in correct path
#SBATCH --output=/home/raisch/DeepVO-pytorch/experiments/sample_name/job.%J.resume.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=claudio.raisch@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

DS=KITTI # NOTE fill in DS to copy, options: KITTI, CARLA
EXPERIMENT=sample_name # NOTE fill in experiment name
EPOCHS=250 # NOTE fill in DS to copy, options: KITTI, CARLA
BATCH_SIZE=8 # NOTE fill in DS to copy, options: KITTI, CARLA
START_EP=250 # NOTE fill in start epoch
TRAIN_SEQ="00 01 02 05 08 09" # NOTE fill in start epoch
VALID_SEQ="03 04 06 10" # NOTE fill in start epoch
ids=( "00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" ) # NOTE fill in sequences to copy

HOME=/home/raisch
PROJECT_DIR=${HOME}/DeepVO-pytorch
EXPERIMENT_DIR=${PROJECT_DIR}/experiments/${EXPERIMENT}
DATASET_DIR=${HOME}/Datasets/${DS}

echo "copy dataset to /scratch/${SLURM_JOB_ID}/:"
mkdir /scratch/${SLURM_JOB_ID}/images /scratch/${SLURM_JOB_ID}/poses_gt
for id in "${ids[@]}"
do
     echo "copy sequence ${id}..."
     rsync --info=progress2 -r ${DATASET_DIR}/images/${id} /scratch/${SLURM_JOB_ID}/images
     rsync --info=progress2 ${DATASET_DIR}/poses_gt/${id}.npy /scratch/${SLURM_JOB_ID}/poses_gt
done

echo "copy sourcecode to /scratch/${SLURM_JOB_ID}/:"
files=( "data_helper.py" "utils.py" "train.py" "model.py" )
for file in "${files[@]}"
do
    rsync --info=progress2 ${PROJECT_DIR}/${file} /scratch/${SLURM_JOB_ID}
done

echo "copy configuration to /scratch/${SLURM_JOB_ID}/:"
rsync --info=progress2 ${EXPERIMENT_DIR}/params.py /scratch/${SLURM_JOB_ID}/params.py

echo "list /scratch/${SLURM_JOB_ID}/:"
ls -l /scratch/${SLURM_JOB_ID}
ls -l /scratch/${SLURM_JOB_ID}/images
ls -l /scratch/${SLURM_JOB_ID}/poses_gt
du -sh /scratch/${SLURM_JOB_ID}/images

####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

IMG_FILE=/common/singularityImages/TCML-Cuda10_1_TF2_1_0_PT1_4.simg

# NOTE check if call is as desired
echo "execute training..."
singularity exec --nv ${IMG_FILE} python3 /scratch/${SLURM_JOB_ID}/train.py ${EXPERIMENT_DIR} /scratch/${SLURM_JOB_ID} ${EPOCHS} ${BATCH_SIZE} ${TRAIN_SEQ} -vseq ${VALID_SEQ} -log ${PROJECT_DIR}/logs -run ${EXPERIMENT} --resume -load_model ${EXPERIMENT_DIR}/${EXPERIMENT}_valid.model -load_optim ${EXPERIMENT_DIR}/${EXPERIMENT}_valid.optimizer -ep ${START_EP}
echo " done!"

