# Universality Random Feature Models
** This repository contains code for reproducing the experiments in our paper "Data-dependent ternarization of random features models: A Random Matrix Theory approach" **
## Content
* Histogram.ipnb
* Random features-based regression experiments with different kernels including our well tuned ternary kernel
* Comparisons generic RelU random features kernel with our proposed well tuned ternary kernel
* Comparisons with LP-RFF methods (The repository of \cite{zhangmay2018lprffs} has been used)
* [Citation](#citation)

## Dependencies
* Python 3.5
* Numpy, scipy, matplotlib, Tensorflow (For MNIST/Fashion-MNIST dataset), Pytorch (for LP-RFF)
## Simulations
* To reproduce Figure 1 in our paper, run
## Citation
```
@article{zhangmay2018lprffs,
  title={Low-Precision Random Fourier Features for Memory-Constrained Kernel Approximation},
  author={Zhang, Jian and May, Avner and Dao, Tri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:1811.00155},
  year={2018}
}
