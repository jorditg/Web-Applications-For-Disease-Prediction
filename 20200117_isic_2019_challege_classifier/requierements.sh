#!/bin/bash

pip install efficientnet-pytorch
pip install isic-challenge-scoring
pip install albumentations
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

