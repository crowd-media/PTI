# Talking Heads AI

This repository contains all services, tools and environments related to the PTI project
## Setup

### Install dependencies

Ffmpeg is required to write videos. If using Ubuntu, it can be installed using the following commands.

```shell
sudo apt-get update
sudo apt-get install libsndfile1-dev
sudo apt-get install ffmpeg
```

**[Optional]** Create conda environment with python 3.10.14

```shell
conda create -n PTI python=3.10.14
conda activate PTI
```

Install poetry and the project dependencies
```shell
pip install poetry
poetry install
```