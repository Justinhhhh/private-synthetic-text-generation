# Private Synthetic Text Generation with Diffusion Models
[![Arxiv](https://img.shields.io/badge/Arxiv-2410.22971-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.22971)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-3819/)
[![Anaconda 24.3.0](https://img.shields.io/badge/Anaconda-24.3.0-green)](https://anaconda.org/anaconda/conda/files?sort=time&sort_order=desc&type=&version=24.3.0)
[![PyTorch 2.3.0](https://img.shields.io/badge/PyTorch-2.3.0-lightgrey)](https://pytorch.org/get-started/previous-versions/#v230)

## Description

This repository contains the source code to replicate the experimental results in our paper.

## Installation

We use [Anaconda 24.3.0](https://anaconda.org/anaconda/conda/files?sort=time&sort_order=desc&type=&version=24.3.0) to set up our virtual environment in Python.
```bash
conda create -n private-synthetic-text-generation python=3.8
conda activate private-synthetic-text-generation
```
We install the remaining requirements with pip.
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data
Please download the respective datasets and put the csv files in the destination folders (SWMH access needs to be granted by its creators).

| Dataset   | Source                                                                         | Manually move to |
|-----------|--------------------------------------------------------------------------------|------------------|
| Drugs.com | Already in Repository                                                          | not needed       |
| SPAM      | [🔗](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) | data/spam/ 📂    |
| SWMH      | [🔗](https://zenodo.org/records/6476179)                                       | data/swmh/ 📂    |
| Thumbs-Up | Already available on huggingface datasets                                      | not needed       |
| WebMD     | [🔗](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset) | data/webmd/ 📂   |

Then you can run the three preprocessing script:
```bash
python preprocessing.py
python create_samples.py
python create_val_sets.py
```

## Pretrained Models

Our code relies on some publicly available text diffusion model checkpoints, which you can download here:  

| Model       | Source                                                                                      | Manually move to |  
|-------------|---------------------------------------------------------------------------------------------|------------------|  
| GENIE       | [🔗](https://drive.google.com/file/d/1-AZssEmgs0QdTp_w8-_4cPi0cV-Hot4N/view)                | GENIE/ 📂        |
| DiffuSeq    | [🔗](https://drive.google.com/file/d/1gj9OpGlM9OzbbrCIOfia8Ve6GMDd2Vxa/view?usp=drive_link) | DiffuSeq/ 📂     |
| SeqDiffuSeq | t.b.d.                                                                                      | SeqDiffuSeq/ 📂  |

## Cite

Please use the following citation:

```
@misc{ochs2024privatesynthetictextgeneration,
      title={Private Synthetic Text Generation with Diffusion Models}, 
      author={Sebastian Ochs and Ivan Habernal},
      year={2024},
      eprint={2410.22971},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.22971}, 
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
