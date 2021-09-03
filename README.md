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
* To reproduce Figure 1 in our paper
** For our TRF method , run (Change n_feat accordingly)
```
python run_model.py   --approx_type=ternary  --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For Nystrom, run (Change n_feat accordingly)
```
python run_model.py   --approx_type=nystrom --do_fp_feat --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For LP-RFF (1 bit) , run (Change n_feat accordingly)
```
python run_model.py   --approx_type=cir_rff --n_bit_feat 1 --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For LP-RFF (8 bit) , run (Change n_feat accordingly)
```
python run_model.py   --approx_type=cir_rff --n_bit_feat 8 --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```

## Citation
@article{zhangmay2018lprffs,
  title={Low-Precision Random Fourier Features for Memory-Constrained Kernel Approximation},
  author={Zhang, Jian and May, Avner and Dao, Tri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:1811.00155},
  year={2018}
}
