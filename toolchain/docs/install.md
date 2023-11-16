# TurnkeyML Installation Guide

The following describes how to install TurnkeyML.

## Operating System Requirements

This project is tested using `ubuntu-latest` and `windows-latest`. `ubuntu-latest` currently defaults to Ubuntu 20.04. However, it is also expected to work on other recent versions of Ubuntu (e.g. 18.04, 22.04), and other flavors of Linux (e.g. CentOS).

## Installing TurnkeyML

### Step 1: Miniconda environment setup

We hughly recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) environments when:

If you are installing TurnkeyML on **Linux**, simply run the command below:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

If you are installing TurnkeyML on **Windows**, manually download and install [Miniconda3 for Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). Please note that PowerShell is recommended when using miniconda on Windows.

Then create and activate a virtual environment like this:

```
conda create -n tkml python=3.8
conda activate tkml
```

### Step 2: Installing TurnkeyML from source

First, make sure you have a copy of the repository locally:

```
git clone https://github.com/aig-bench/onnxmodelzoo.git
```

Then, simply pip install the TurnkeyML package:

```
pip install -e onnxmodelzoo/toolchain
```

You are now done installing TurnkeyML! 

If you are planning to use the `turnkey` tools with the TurnkeyML models or Slurm please see the corresponding sections below.

## TurnkeyML Models Requirements

The TurnkeyML models are located at `install_path/toolchain/models`, which we refer to as `models/` in most of the guides.

> _Note_: The `turnkey models location` command and `turnkey.common.filesystem.MODELS_DIR` are useful ways to locate the `models` directory. If you perform PyPI installation, we recommend that you take an additional step like this:

```
(tkml) jfowers:~$ turnkey models location

Info: The TurnkeyML models directory is: ~/onnxmodelzoo/toolchain/models
(tkml) jfowers:~$ export models=~/onnxmodelzoo/toolchain/models
```

The `turnkeyml` package only requires the packages to run the tools. If you want to run the models as well, you will also have to install the models' requirements. 

In your `miniconda` environment:

```
pip install -r models/requirements.txt
```

## (Optional) Installing Slurm support

Slurm is an open source workload manager for clusters. If you would like to use Slurm to build multiple models simultaneously, please follow the instructions below.

### Setup your Slurm environment

Ensure that your onnxmodelzoo folder and your conda installation are both inside a shared volume that can be accessed by Slurm.
Then, run the following command and wait for the Slurm job to finish:

```
sbatch --mem=128000 toolchain/src/turnkeyml/cli/setup_venv.sh
```

### Get an API token from Huggingface.co (optional)

Some models from Huggingface.co might require the use of an API token. You can find your api token under Settings from your Hugging Face account.

To allow slurm to use your api token, simply export your token as an environment variable as shown below:


```
export HUGGINGFACE_API_KEY=<YOUR_API_KEY>
```

### Setup a shared download folder (optional)

Both Torch Hub and Hugging Face models save model content to a local cache. A good practice is to store that data with users that might use the same models using a shared folder. TurnkeyML allows you to setup a shared ML download cache folder when using Slurm by exporting an environment variable as shown below:


```
export SLURM_ML_CACHE=<PATH_TO_A_SHARED_FOLDER>
```

### Test it

Go to the `models/` folder and build multiple models simultaneously using Slurm.

```
turnkey selftest/*.py --use-slurm --build-only --cache-dir PATH_TO_A_CACHE_DIR
```