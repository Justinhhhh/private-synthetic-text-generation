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
conda create -n private-synth-textgen python=3.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
We install the remaining requirements with pip.
```bash
pip install -r requirements.txt
```

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

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. # private-synthetic-text-generation
TODO
