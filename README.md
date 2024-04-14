[![DOI](http://img.shields.io/badge/DOI-TO.ADD.DATASET-blue)](https://TO.BE.UPDATED.SOON)
[![DOI](http://img.shields.io/badge/DOI-TO.ADD.MODELS-blue)](https://TO.BE.UPDATED.SOON)
[![DOI](http://img.shields.io/badge/DOI-TO.ADD.PAPER-B31B1B)](https://TO.BE.UPDATED.SOON)

# Segmentation and analysis of OCT images

<a name="contents"></a>
## 📖 Contents
- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Data Access](#data-access)
- [How to Cite](#how-to-cite)

<a name="introduction"></a>
## 🎯 Introduction - TO BE UPDATED SOON

<a name="data"></a>
## 📁 Data - TO BE UPDATED SOON

|  ![Source image](.assets/source_img.png "Source image")  |  ![Pre-processed image](.assets/gray_img.png "Pre-processed image")  |
|:------------------------------------------------------:|:------------------------------------------------------------------:|
|                     *Source image*                     |                       *Pre-processed image*                        |

<a name="methods"></a>
## 🔬 Methods - TO BE UPDATED SOON

<a name="results"></a>
## 📈 Results - TO BE UPDATED SOON

<a name="conclusion"></a>
## 🏁 Conclusion - TO BE UPDATED SOON

<a name="requirements"></a>
## 💻 Requirements

- Operating System
  - [x] macOS
  - [x] Linux
  - [x] Windows (limited testing carried out)
- Python 3.11.x
- Required core libraries: [environment.yaml](environment.yaml)

<a name="installation"></a>
## ⚙ Installation

**Step 1: Install Miniconda**

Installation guide: https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

**Step 2: Clone the repository and change the current working directory**
``` bash
git clone https://github.com/ViacheslavDanilov/oct_segmentation.git
cd oct_segmentation
```

**Step 3: Set up an environment and install the necessary packages**
``` bash
chmod +x make_env.sh
./make_env.sh
```

<a name="how-to-run"></a>
## 🚀 How to Run - TO BE UPDATED SOON

Specify the `data_path` and `save_dir` parameters in the [predict.yaml](configs/predict.yaml) configuration file. By default, all images within the specified `data_path` will be processed and saved to the `save_dir` directory.

To run the pipeline, execute [predict.py](src/models/smp/predict.py) from your IDE or command prompt with:
``` bash
python src/models/smp/predict.py
```

<a name="data-access"></a>
## 🔐 Data Access - TO BE UPDATED SOON
All essential components of the study, including the curated dataset and trained models, have been made publicly available:
- **Dataset:** [https://zenodo.org](https://zenodo.org)
- **Models:** [https://zenodo.org](https://zenodo.org)

<a name="how-to-cite"></a>
## 🖊️ How to Cite - TO BE UPDATED SOON
Please cite [OUR PAPER](https://TO.BE.UPDATED.SOON) if you found our data, methods, or results helpful for your research:

> Danilov V.V., Laptev V.V., Klyshnikov K.Yu., Ovcharenko E.A. (**2024**). _PAPER TITLE_. **Journal Title**. DOI: [TO.BE.UPDATED.SOON](TO.BE.UPDATED.SOON)
