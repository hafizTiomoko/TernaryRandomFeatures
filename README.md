## Dependencies
We run our code with the following dependencies: Python 3.5, Numpy 1.9.0, scikit-learn 0.15.2, matplotlib 1.4.0, Tensorflow (For MNIST/Fashion-MNIST dataset), torchvision 0.2.0
## Simulations
* TRF.ipynb contains the necessary ingredients to run our TRF method given some normalized data
* To reproduce Figure 1 in our paper <br />
** Follow the steps in the folder Comparisons_LP_RFFs/lp_rffs (basecode from \cite{zhangmay2018lprffs}), to install the necessary packages.
** For our TRF method , run (Change n_feat, the number of random features,  accordingly)
```
python run_model.py   --approx_type=ternary  --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For Nystrom, run (Change n_feat, the number of random features, accordingly)
```
python run_model.py   --approx_type=nystrom --do_fp_feat --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For LP-RFF (1 bit) , run (Change n_feat, the number of random features, accordingly)
```
python run_model.py   --approx_type=cir_rff --n_bit_feat 1 --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```
** For LP-RFF (8 bit) , run (Change n_feat, the number of random features, accordingly)
```
python run_model.py   --approx_type=cir_rff --n_bit_feat 8 --n_feat=50000  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05    --epoch=10 --learning_rate=1 --fixed_epoch_number   --kernel_sigma=0.9128709291752769 --random_seed=2   --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
```

* To reproduce Figure 2, use Histograms_kernels.ipynb
* To reproduce Figure 3, use the Jupyter notebook Sparse-Random-Features-based-ridge-regression.ipynb
* To reproduce Figure 4 in our paper <br />
** For our TRF method, run (Change n_feat, the number of random features, accordingly)
```
python run_model.py   --approx_type=ternary --n_feat=50000  --opt=sgd --learning_rate=0.001 --n_bit_feat=1    --model=ridge_regression --l2_reg=0.005 --kernel_sigma=28.867513459481287 --random_seed=1   --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample
```
** For the LP-RFF (8 bit) run (Change n_feat, the number of random features, accordingly) - Change --n_bit_feat=8 to --n_bit_feat=1 for LP-RFF (1 bit)
```
python run_model.py   --approx_type=rff --n_feat=5000  --n_bit_feat=8    --model=ridge_regression --l2_reg=0.05 --kernel_sigma=28.867513459481287 --random_seed=1   --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample
```
** For Nystrom, run (Change n_feat, the number of random features, accordingly) 
```
python run_model.py   --approx_type=nystrom --n_feat=10000 --n_bit_feat=8    --model=ridge_regression --l2_reg=0.05 --kernel_sigma=28.867513459481287 --random_seed=1   --data_path=../LP-RFFs-Data/census --save_path=./tmp  --collect_sample_metrics
```
## Citation
@article{zhangmay2018lprffs,
  title={Low-Precision Random Fourier Features for Memory-Constrained Kernel Approximation},
  author={Zhang, Jian and May, Avner and Dao, Tri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:1811.00155},
  year={2018}
}
