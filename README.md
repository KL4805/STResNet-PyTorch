# ST-ResNet-Pytorch: Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction

## Introduction

This repo contains a PyTorch Implementation of ST-ResNet. 

Paper: Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction, Junbo Zhang, Yu Zheng, Dekang Qi, In AAAI, 2017. 

Acknowledgement: Many of the preprocessing code is taken from https://github.com/TolicWang/DeepST, which is a TensorFlow Implementation. 

## Data

We provide TaxiBJ dataset, which can be downloaded from the link given in https://github.com/TolicWang/DeepST/issues/3. 

After downloading, copy the data under `data/TaxiBJ`. 

## Dependency

PyTorch 1.6.0

python 3.8.3

numpy

h5py

## Usage

Go to the path `TaxiBJ`, set parameters in `exprTaxiBJ.py`, and run `python exprTaxjBJ.py`. 

## Re-implementation Results

We report some of the results achieved by our code, as well as what the paper claims. 

(By default, we use external features, use batch normalization, and use parameterized fusion. La-Cb-Pc-Td denotes ST-ResNet with $a$ residual blocks, $b$ closeness time steps, $c$ period time steps and $d$ trend time steps. )

| Setting\Results (RMSE) | Ours  | Paper         |
| ---------------------- | ----- | ------------- |
| L4-C3-P1-T1            | 17.98 | 17.51 (No BN) |
| L6-C3-P1-T1            | 17.88 | /             |
| L12-C3-P1-T1           | 17.65 | 16.69         |

