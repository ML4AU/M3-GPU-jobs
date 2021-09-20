# Running single GPU jobs on MASSIVE M3

Before running a multi-GPU job on the cluster, it's always beneficial to run
a single GPU job first, so you can compare results - sometimes adding more
GPUs can slow your code down. In particular, when using powerful GPUs
like NVIDIA Tesla V100s on small training sets, added GPUs are unlikely
to offer any benefits, or even introduce additional time with 
communication overhead between the GPUs.

In this section, we'll start with a GUI interface using a
[Strudel Desktop](https://beta.desktop.cvl.org.au/login)
session to ensure the code is behaving as expected, 
and then translate that into an sbatch job submission script. 
This section assumes you have cloned this repository and the 
Tensorflow models repositroy, and have set up a Python environment, per
the README file in the parent directory.

# Starting with a desktop
## Start a desktop session
To get started, we'll want to get a desktop session with a GPU attached.
You can find more information including images on how to complete these steps in our 
[documentation here](https://docs.massive.org.au/M3/connecting/strudel2/connecting-to-desktop.html?highlight=strudel2),
but the interface should be straightforward without them. A desktop is the easiest and 
fastest way to get access to a GPU for interactive testing.

1. Navigate to https://beta.desktop.cvl.org.au/login and login with the 
   email attached to your MASSIVE account. 
2. Once logged in, select the `Desktop` tab on the left hand side of the page.
3. In the `GPUs` dropdown menu, select P4 - a K1 desktop isn't sufficient
   for machine learning, and the K80s, while sufficient, tend to have a longer
   wait time attached to them and don't perform as well for machine learning
   workloads. 
4. Ensure in the `Runtime` section, you request an hour - this will be more 
   than enough time to run our basic examples, without waiting an extended time
   for access to the P4 GPU. 
5. Select `Launch`, and wait for your desktop to start under `Pending/Running Desktops`. 
   Once it appears, select `Connect`. Your desktop will open in a new tab, assuming
   that you don't have pop ups blocked. 
6. Open a terminal in the desktop - here is where we'll test our commands. You will want 
   to have a second terminal open to, so we can use the `nvidia-smi` command to verify
   our GPU is utilised later. 

## Let's run a model! 
The commands here will be ran inside of the terminal you opened in your desktop session,
and are based on instructions in the 
[Tensorflow models repository.](https://github.com/tensorflow/models)

```
# Replace <project> with your project ID, i.e. ab12
[user@m3p001 ~] export REPODIR=/scratch/<projects>/$USER/gpu-examples/

# Activate your conda environment
[user@m3p001 ~] source /path/to/miniconda/bin/activate
(base)[user@m3p001 ~] conda activate tf2-gpu

# Add the Tensorflow models to our Python path
(tf2-gpu)[user@m3p001] export PYTHONPATH=${REPODIR}/models:$PYTHONPATH

# Set your variables for your model and data directories
# And the number of GPUs, which is 1 here
(tf2-gpu)[user@m3p001] export DATA_DIR=${REPODIR}/M3-GPU-jobs/mnist-data
(tf2-gpu)[user@m3p001] export MODEL_DIR=${REPODIR}/M3-GPU-jobs/single-gpu-examples/desktop-mnist
(tf2-gpu)[user@m3p001] export NUM_GPU=1

# In a separate terminal, run nvidia-smi to watch GPU utilisation.
# This should show 0% now, but will change we we run the next command

watch nvidia-smi
Every 2.0s: nvidia-smi                                                                                                                                                Thu Sep 16 11:44:42 2021

Thu Sep 16 11:44:42 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P4            On   | 00000000:00:0D.0 Off |                    0 |
| N/A   28C    P8     6W /  75W |     15MiB /  7611MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     36208	 G   X                                  15MiB |
+-----------------------------------------------------------------------------+


# Run the included MNIST model in the terminal which isn't running nvidia-smi
# Note the --distribution_strategy flag to determine how many GPUs are used
# We're using a single GPU, so this will be one_device
(tf2-gpu)[user@m3p001 ~] python ${REPODIR}/models/official/vision/image_classification/mnist_main.py \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --train_epochs=10 \
    --distribution_strategy=one_device \
    --num_gpus=$NUM_GPU \
    --download 
# At the end of the successful run, you'll have some stats like this
Run stats:
{'accuracy_top_1': 0.9731987714767456, 'eval_loss': 0.08816861692402098, 'loss': 0.11313708409153182, 'training_accuracy_top_1': 0.9655340909957886} 

# Note, you only need to run with the --download flag once, this downloads 
# the MNIST dataset to our $DATA_DIR

```

If you watch the terminal with `watch nvidia-smi` running, you can see your
GPU utilisation increase. 

Now we can try to run something a bit more intensive, the RESNET model 
on some CIFAR-10 data. Some of these steps are repeats of steps ran above,
so if you have already activated your conda environment and exported
you `PYTHONPATH`, there's no need to repeat the steps here. 


```
# Replace <project> with your project ID, i.e. ab12
[user@m3p001 ~] export REPODIR=/scratch/<projects>/$USER/gpu-examples/

# Activate your conda environment
[user@m3p001 ~] source /path/to/miniconda/bin/activate
(base)[user@m3p001 ~] conda activate tf2-gpu

# Add the Tensorflow models to our Python path
(tf2-gpu)[user@m3p001] export PYTHONPATH=${REPODIR}/models:$PYTHONPATH

# Set your variables for your model and data directories
# And the number of GPUs, which is 1 here
(tf2-gpu)[user@m3p001] export MODEL_DIR=${REPODIR}/M3-GPU-jobs/single-gpu-examples/desktop-resnet
(tf2-gpu)[user@m3p001] export DATA_DIR=${REPODIR}/M3-GPU-jobs/cifar-data
(tf2-gpu)[user@m3p001] export NUM_GPU=1

# Extract the CIFAR-10 data into your DATA_DIR
(tf2-gpu)[user@m3p001] tar xf $DATA_DIR/cifar-10-binary.tar.gz

# In a separate terminal, run nvidia-smi to watch GPU utilisation if you want to 
# monitor it.
# Then, run the RESNET model on your CIFAR-10 data.
# Note, --distribution_strategy=one_device as we're using one GPU
(tf2-gpu)[user@m3p001] python ${REPODIR}/models/official/vision/image_classification/resnet_cifar_main.py  \
    --num_gpus=$NUM_GPU  \  
    --batch_size=128  \   
    --model_dir=$MODEL_DIR \    
    --data_dir=$DATA_DIR/cifar-10-batches-bin \
    --distribution_strategy=one_device

# At the end of the successful run, you'll have some stats like this
I0917 12:01:09.244876 139914427299648 keras_utils.py:88] BenchmarkMetric: {'global step':100, 'time_taken': 16.552601,'examples_per_second': 773.292362}
I0917 12:01:18.520732 139914427299648 keras_utils.py:88] BenchmarkMetric: {'global step':200, 'time_taken': 9.275636,'examples_per_second': 1379.959218}
I0917 12:01:27.772545 139914427299648 keras_utils.py:88] BenchmarkMetric: {'global step':300, 'time_taken': 9.252051,'examples_per_second': 1383.477080}
I0917 12:01:36.087600 139914427299648 keras_utils.py:96] BenchmarkMetric: {'epoch':0, 'time_taken': 43.452109}
390/390 - 43s - loss: 2.7444 - categorical_accuracy: 0.2385
78/78 - 3s - loss: 3.2528 - categorical_accuracy: 0.1654
{'accuracy_top_1': 0.1653645783662796, 'eval_loss': 3.2527751739208517, 'loss': 2.744446820479173, 'training_accuracy_top_1': 0.23846153914928436, 'step_timestamp_log': ['BatchTimestamp<batch_index: 1, timestamp: 1631844052.6918662>', 'BatchTimestamp<batch_index: 100, timestamp: 1631844069.2444673>', 'BatchTimestamp<batch_index: 200, timestamp: 1631844078.5201037>', 'BatchTimestamp<batch_index: 300, timestamp: 1631844087.7721543>'], 'train_finish_time': 1631844096.0887015, 'avg_exp_per_second': 1094.6318289056542}
```

# Submitting a single-GPU job to the cluster
So far, we've been interactively running commands in the desktop terminal. 
Once you know everything works, you'll likely want to transition to a 
job suibmission script. This will allow you to get access to higher powered GPUs,
and allow you to reprocude your code with a single job submission, rather
than remembering to run commands every time.

You can inspect the sample jobs for these in 
`/path/to/gpu-examples/M3-GPU-jobs/single-gpu-examples/` where you
will find `mnist_job.bash` and `resnet_job.bash`.

```
cat mnist_job.bash

#!/bin/bash
#SBATCH --job-name=mnist_job

# Replace this with your email address
# To get email updates when your job starts, ends, or fails
#SBATCH --mail-user=youremail@domain.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Replace <project> with your project ID
#SBATCH --account=<project>

#SBATCH --time=00:30:00
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
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
export DATA_DIR=${REPODIR}/M3-GPU-jobs/mnist-data
export MODEL_DIR=${REPODIR}/M3-GPU-jobs/single-gpu-examples/job-mnist
export NUM_GPU=1

# Run the included MNIST model
# Note the --distribution_strategy flag to determine how many GPUs are used
# We're using a single GPU, so this will be one_device

python ${REPODIR}/models/official/vision/image_classification/mnist_main.py \
    --model_dir=$MODEL_DIR \
    --data_dir=$DATA_DIR \
    --train_epochs=10 \
    --distribution_strategy=one_device \
    --num_gpus=$NUM_GPU \
EOF
```

You will note the commands we ran on the command line are replicated
here, with some commands at the top which communicate the resources we need to 
the SLURM scheduler with `#SBATCH`. We start the script off with the shebang, 
indicating the script is written in bash. Then we specify some optional
parameters, like our job name, and an email address to recieve updates on our job status.
GPU jobs often wait a while in the queue, so getting an email notification when they start
and end can be helpful.

The other included parameters are more important - your HPC project, and the amount of time required 
(which is 30 minutes here). The `ntasks` specifies number of CPUs, and the `mem` specifies the 
RAM or memory - both of these values are based on the default P4 job, which we 
already ran successfully. Importantly, `gres=gpu:1` specifies we want one GPU, and we also 
request the m3h partition - this ensures we request a job on the partition with P100
GPUs. If we request a partition without a GPU our job will fail due to invalid combination
of resources. 

If you're new to submitting scripts to a scheduler with SLURM, note that your `#SBATCH`
commands must be at the top of the script without interruption between them (you can't module load
between #SBATCH commands), and the formatting is fairly strict - `# SBATCH` with a space won't work. 

You can submit these scripts with 

```
sbatch mnist_job
sbatch resnet_job
```

You can query your jobs with the `squeue` and `sacct` command, and find the 
outputs of your jobs in `slurm-<jobID>.out` and `slurm<jobID>.err` files.
