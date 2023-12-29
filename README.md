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
```

### Environment Setup

```shell
conda create -n cdul python=3.10.12
conda activate cdul
pip install -r requirements.txt
```
