# Segmentation and analysis of OCT images
This repository is dedicated to the segmentation of [optical coherence tomography](https://en.wikipedia.org/wiki/Optical_coherence_tomography) (OCT) images and the analysis of the plaques that appear on them.

## Requirements

- Linux or macOS (Windows has not been officially tested)
- Python 3.8.x

## Installation

Step 1: Download and install Miniconda
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Step 2: Clone the repository, create a conda environment and install the requirements for the repository
``` bash
git clone https://github.com/ViacheslavDanilov/oct_segmentation.git
cd oct_segmentation
chmod +x create_env.sh
./create_env.sh
pip install -r requirements.txt --no-cache-dir
```

Step 3: Initialize the git hooks using the pre-commit framework
``` bash
pre-commit install
```

Step 4 (Optional): Download a raw dataset using DVC
``` bash
dvc pull dvc/data/raw.dvc
```
NOTE: Since data storage is organized through Google Drive, you should first request access to the [DVC repository](https://drive.google.com/drive/folders/1SnUrqIBS60UR5l9HUMCiyxFFeBPM7LL4?usp=share_link).

## Data

|  ![Source image](media/source_img.png "Source image")  |  ![Pre-processed image](media/gray_img.png "Pre-processed image")  |
|:------------------------------------------------------:|:------------------------------------------------------------------:|
|                     *Source image*                     |                       *Pre-processed image*                        |
