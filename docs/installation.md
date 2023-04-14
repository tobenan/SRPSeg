## Installation
The code is developed using Python 3.8 with PyTorch 1.8.0. The code is developed and tested using 2 3090 GPUs.

1. **Clone this repo.**

   ```shell
   $ git clone https://github.com/tobenan/MIPSeg.git
   $ cd MIPSeg
   ```

2. **Install dependencies.**

   **(1) Create a conda environment:**

   ```shell
   $ conda env create -f MIPSeg.yaml
   $ conda activate MIPSeg
   ```

   **(2) Install apex 0.1(needs CUDA) in furnace**

   ```shell
   $ cd ./furnace/apex
   $ python setup.py install --cpp_ext --cuda_ext
   ```
  
## Optional
We recommend using docker to run experiments. Here is the docker name: charlescxk/ssc:2.0 .
You could pull it from https://hub.docker.com/ and mount your local file system to it.
