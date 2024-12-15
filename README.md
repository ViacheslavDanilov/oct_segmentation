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
## 🎯 Introduction
This repository provides a comprehensive approach for deep learning-based segmentation and quantification of atherosclerotic plaque features in optical coherence tomography ([OCT](https://en.wikipedia.org/wiki/Optical_coherence_tomography)) images. The accurate analysis of plaques is critical for preventing cardiovascular events and guiding therapeutic interventions. By leveraging state-of-the-art deep learning models, this project enables precise identification of lumen, fibrous cap, lipid core, and vasa vasorum features, contributing to advancements in cardiovascular diagnostics.

<a name="data"></a>
## 📁 Data
The dataset comprises OCT images from 103 patients, collected across multiple centers. These images include 25,698 annotated slices, detailing four key plaque features:

- **Lumen (LM):** Vascular opening
- **Fibrous Cap (FC):** Thin protective layer over lipid core
- **Lipid Core (LC)**: Lipid-rich region associated with vulnerable plaques
- **Vasa Vasorum (VV)**: Microvessels supplying the arterial wall

Annotations were performed by cardiologists using [Supervisely](https://supervisely.com/), with double-verification for accuracy (refer to <a href="#figure-1">Figure 1</a>). The dataset is structured for 5-fold cross-validation to ensure robust model evaluation. For more details, refer to the Dataset Repository at [https://doi.org/10.5281/zenodo.14478209](https://doi.org/10.5281/zenodo.14478209).

<p align="center">
  <img id="figure-1" width="100%" height="100%" src=".assets/annotation_methodology.png" alt="Annotation methodology">
</p>

<p align="left">
    <em><strong>Figure 1.</strong> Annotation methodology for optical coherence tomography images depicting plaque morphological features associated with atherosclerotic plaque development. The feature annotations delineated with segmentation masks include the lumen (pink), fibrous cap (blue), lipid core (blue), and vasa vasorum (red).</em>
</p>

<a name="methods"></a>
## 🔬 Methods
We employed nine advanced deep learning architectures — [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), [U-Net++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1), [DeepLabV3](https://arxiv.org/abs/1706.05587), [DeepLabV3+](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49), [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf), [LinkNet](https://ieeexplore.ieee.org/document/8305148), [PSPNet](https://arxiv.org/abs/1612.01105), [PAN](https://arxiv.org/abs/1805.10180), and [MA-Net](https://ieeexplore.ieee.org/document/9201310) — to address the segmentation of atherosclerotic plaque features. To enhance performance, we applied a hybrid segmentation strategy, which involves designing specialized models for each plaque feature based on their prevalence and complexity:
- **Lumen Segmentation:** A single-class model trained exclusively for the lumen, leveraging its high representation in the dataset.
- **Fibrous Cap and Lipid Core Segmentation:** A two-class model trained jointly on these features due to their shared morphological characteristics and moderate dataset representation.
- **Vasa Vasorum Segmentation:** A separate single-class model focused on this rare feature to better capture its unique characteristics.

Each model underwent a comprehensive tuning process to optimize its performance, which included:
- **Hyperparameter Tuning:** We employed Bayesian optimization, combined with the HyperBand early stopping strategy, to explore different configurations and identify the most effective hyperparameters for each model.
- **Data Augmentation:** To enhance the robustness of the models and prevent overfitting, various augmentation techniques were applied, such as random brightness adjustments, rotations, and scaling.
- **Cross-Validation:** A 5-fold cross-validation strategy was used to partition the data, ensuring that each model was trained and evaluated on different subsets of the dataset, which prevented data leakage and helped assess model generalizability.

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
