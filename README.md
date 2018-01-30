# MSDNet - reproducibility and applying GCN blocks with separable kernel

This repository contains a reproduction code
(in PyTorch) for
"[MSDNet: Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/abs/1703.09844)"

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)

## Introduction

MSDNet is a novel approach fo image classification with computational resource limits
at test time. This repository provides an implementation based on the technical description
provided in the paper. Currently this code implements the support for Cifar-10 and Cifar-100.

Moreover, this code integrates the support for GCN based layers instead of normal convolution layers,
in order to reduce the model parameters.

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(0.1.12+)](http://pytorch.org)


### Train
As an example, use the following command to train an MSDNet on Cifar10

```
python3 main.py --model msdnet -b 64 -j 2 cifar10 --msd-blocks 10 --msd-base 4 \
--msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0
```

As an example, use the following command to train an MSDNet on Cifar100 with GCN block

```
python3 main.py --model msdnet -b 64 -j 2 cifar100 --msd-blocks 10 --msd-base 3 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn
```


### Evaluation
We take the Cifar10 model trained above as an example.

To evaluate the trained model, use `evaluate` to evaluate from the default checkpoint directory:

```
python3 main.py --model msdnet -b 64 -j 2 cifar100 --msd-blocks 10 --msd-base 3 \
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 \
--msd-share-weights --msd-all-gcn --resume --evaluate
```


### Other Options
For detailed options, please `python main.py --help`

For more examples and using pre-trained models, please `less script.sh`
