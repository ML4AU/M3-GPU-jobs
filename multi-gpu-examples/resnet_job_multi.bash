#!/bin/bash
#SBATCH --job-name=multi_gpu_resnet_job

# Replace this with your email address
# To get email updates when your job starts, ends, or fails
#SBATCH --mail-user=<youremail@domain.org>
#SBATCH --mail-type=BEGIN,END,FAIL

# Replace <project> with your project ID
#SBATCH --account=<project>

#SBATCH --time=00:30:00
#SBATCH --ntasks=6
#SBATCH --gres=gpu:2
#SBATCH --partition=m3h
#SBATCH --mem=55G

# module load CUDA and cudNN for GPU access
module load cuda/10.1
module load cudnn/7.6.5.32-cuda10

# Edit this section to activate your conda environment
source /path/to/miniconda/bin/activate
conda activate tf2-gpu

# Edit this to point to your repositories location
export REPODIR=/scratch/<project>/$USER/gpu-examples

# Ensure you're using the TF 2.1.0 version of the models repository
cd $REPODIR/models
git checkout v2.1.0

# Export variables as necessary
# We're using one GPU
export PYTHONPATH=${REPODIR}/models:$PYTHONPATH
export DATA_DIR=${REPODIR}/M3-GPU-jobs/cifar10-data
export MODEL_DIR=${REPODIR}/M3-GPU-jobs/multi-gpu-examples/multi-job-resnet
export NUM_GPU=2

# Run the included Resnet model 
# Note the --distribution_strategy flag to determine how many GPUs are used
# We're using a multiple GPUs, so this will be mirrored

python ${REPODIR}/models/official/vision/image_classification/resnet_cifar_main.py  \
    --num_gpus=$NUM_GPU  \
    --batch_size=128  \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR/cifar-10-batches-bin \
    --distribution_strategy=mirrored \
EOF
