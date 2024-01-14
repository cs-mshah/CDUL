# CDUL
Implementation to CDUL: CLIP-Driven Unsupervised Learning for Multi-Label Image Classification

## Setup

### Clone Repository

```shell
git clone https://github.com/cs-mshah/CDUL.git
cd CDUL
```

Create a `.env` file at the root of `CDUL` to store environment variables. Set the `DATASETS_ROOT` (for downloading the datasets) and `PROJECT_ROOT` (the path to this directory). Example:
```shell
PROJECT_ROOT='~/projects/CDUL'
DATASETS_ROOT='~/datasets'
WANDB_API_KEY=<your wandb api key>
WANDB_ENTITY=<wandb entity (username)>
WANDB_PROJECT=CDUL
```

Also export the above environment variables by adding them to `.bashrc`/`.zshrc`/by [conda env activation](https://guillaume-martin.github.io/saving-environment-variables-in-conda.html).

### Environment Setup

```shell
conda create -n cdul python=3.10.12
conda activate cdul
pip install -r requirements.txt
```

## Running
For convenience, a `Makefile` has been provided to execute underlying commands for various tasks. Run `make help` for all available commands.