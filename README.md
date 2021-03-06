# Running GPU and Multi GPU Jobs on the MASSIVE M3 cluster

As machine learning and deep learning models continue to grow in size,
and training datasets scale accordingly, there's an increased need to take
advantage of GPU technology to accelerate model training. Researchers 
are turning to HPC facilities to get access to advanced GPU hardware. 

On MASSIVE, we provide interactive GPU access via our 
[Strudel Desktop and JupyterLab interface](https://docs.massive.org.au/M3/connecting/connecting-via-strudel2.html). 
This allows researchers to bypass some of the learning curve associated with using HPC so they
can focus on getting their research done. However, these options don't provide 
access to multiple-GPUs, meaning as workloads scale, researchers need to 
request interactive jobs on the command line or use job submission to access these resources.

The aim of this repository is to provide examples for requesting a single GPU and 
multiple GPUs on the command line, with example job submission scripts. In future we intend to add
multi-node examples too.

These instructions reply on the best practice models implemented in the
[TensorFlow models repository](https://github.com/tensorflow/models/tree/master/official).
These instructions are also based on the excellent instructions provided 
in the [Biowulf documentation](https://hpc.nih.gov/docs/deeplearning/multinode_DL.html).

If while looking at these materials you have any additional questions,
don't hesitate to reach out to MASSIVE M3 helpdesk at
help@massive.org.au.

# General comments on GPU jobs on MASSIVE M3
I'm going to spotlight some tips, tricks, general troubleshooting advice, 
and things to note when running GPU jobs on M3 here. 

* Ensure your code is GPU enabled - if it isn't, the GPUs won't accelerate your code!
  You can check this while code is running by using a desktop job and running the `nvidia-smi` command to
  confirm the GPU is being utilised. 
  [TensorFlow also provides methods to check this.](https://www.tensorflow.org/guide/gpu)
* Additionally, if you run a GPU job on a login node or node without a GPU on it, your
  GPU enabled code won't benefit. It seems obvious, but if you're troubleshooting  
  unexpected job behaviour, it's worth double checking you have access to a GPU. 
* Ensure your environment has CUDA and cudNN if required. These can be accessed via modules
  or installed into a conda environment. 
* Ensure you request reasonable resources - if you request 5 GPUs on a node that only has 2,
  or request a GPU and 50 CPUs on a node without 50 CPUs, your job won't start. Check
  [our GPU documentation](https://docs.massive.org.au/M3/GPU-docs/GPU-look-up-tables.html) 
  for more information on this. 

With all of that said, let's run some GPU jobs.

# Set-Up
## Setting up your Python environment
These instructions expect that you will set up a Miniconda environment per the 
[Miniconda Instructions in our documentation.](https://docs.massive.org.au/M3/software/pythonandconda/python-miniconda.html#python-miniconda)
If you already have a Miniconda install, you'll just need to create a new environment
to run the examples. 

```
# Activate the Miniconda environment you installed per the above instructions
source /path/to/miniconda/bin/activate

# Create a new environment to use for these exercises and activate it
conda create --name tf2-gpu
conda activate tf2-gpu

# Install Python, TensorFlow GPU and TensorFlow Datasets into your environment
# TensorFlow may take a while, go make a coffee!
conda install python=3.7 pandas=1.3
pip install tensorflow-gpu==2.1.0

# pip install tensorflow-datasets
pip install tensorflow-datasets==1.2.0
```
You will now have a conda environment set up with everything you need to run the 
examples provided here. 

## Clone the repositories
We'll be relying on code examples from the Tensorflow models repository - we know this code
is appropriately GPU enabled, which will allow us to focus on the HPC component of these exercises.
I recommend doing all of this somewhere in `/scratch`.

```
# Create a directory to store the examples and models
# Replace <project> with your project ID
mkdir -p /scratch/<project>/$USER/gpu-examples

# Store the path in a variable
export REPODIR=/scratch/<project>/$USER/gpu-examples

# Clone this repository
cd $REPODIR
git clone git@github.com:ML4AU/M3-GPU-jobs.git

# Clone the TensorFlow models repository
cd $REPODIR        
git clone https://github.com/tensorflow/models.git

# Switch branches in models to match our TensorFlow version:
cd $REPODIR/models
git checkout v2.1.0

```

## Download the CIFAR-10 data
We will use the CIFAR-10 dataset in our examples, and it needs to be downloaded.

```
cd ${REPODIR}/M3-GPU-jobs/cifar10-data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xf cifar-10-binary.tar.gz
```

# Running The Examples
You will find more detailed instructions and sample job submission scripts in `single-gpu-examples` and `multi-gpu-examples`.

# Future Work
We intend to add multi-node examples to this repository. We're currently working on getting
[Horovod](https://github.com/horovod/horovod) running on the MASSIVE M3 cluster for this purpose. 
