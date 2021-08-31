import numpy as np
import scipy.io as sio
import h5py
import scipy.sparse as sp
from scipy.optimize import least_squares
pi = np.pi

def estim_tau(X):
  tau = np.mean(np.diag(X.T @ X))/np.shape(X)[0]

  return tau


def compute_thresholds(tau, fun):
  if fun is 'rff':
     F = lambda x: ((np.exp(-x[0] ** 2 / tau) + np.exp(-x[1] ** 2 / tau)) / np.sqrt(2 * pi * tau) - np.exp(-tau / 2),
                   (-x[0] * np.exp(-x[0] ** 2 / tau) + x[1] * np.exp(-x[1] ** 2 / tau)) / (
                       np.sqrt(2 * pi * tau ** 3)) - np.exp(-tau / 2) / 2)
    ### relu
  elif fun is 'relu':
    F = lambda x: ((np.exp(-x[0] ** 2 / tau) + np.exp(-x[1] ** 2 / tau)) / np.sqrt(2 * pi * tau) - 1 / 2,
                   (-x[0] * np.exp(-x[0] ** 2 / tau) + x[1] * np.exp(-x[1] ** 2 / tau)) / (
                     np.sqrt(2 * pi * tau ** 3)) - 1 / np.sqrt(8 * pi * tau))

  res = least_squares(F, (1, 1), bounds=((0, 0), (1, 1)))
  return res.x

def load_data(path="../../data/census/census"):
  try:
    X_train = sio.loadmat(path + "_train_feat.mat")
    Y_train = sio.loadmat(path + "_train_lab.mat")
    X_test = sio.loadmat(path + "_heldout_feat.mat")
    Y_test = sio.loadmat(path + "_heldout_lab.mat")
  except:
    print("switch to use h5py to load files")
    X_train = h5py.File(path + "_train_feat.mat", 'r')
    Y_train = sio.loadmat(path + "_train_lab.mat")
    X_test = sio.loadmat(path + "_heldout_feat.mat")
    Y_test = sio.loadmat(path + "_heldout_lab.mat")




  if 'X_ho' in X_test.keys():
    X_test = X_test['X_ho']
  else:
    X_test = X_test["fea"]
  if "X_tr" in X_train.keys():
    X_train = X_train['X_tr']
  else:
    X_train = X_train['fea']
  if "Y_ho" in Y_test.keys():
    Y_test = Y_test['Y_ho']
  else:
    Y_test = Y_test['lab']
  if "Y_tr" in Y_train.keys():
    Y_train = Y_train['Y_tr']
  else:
    Y_train = Y_train['lab']

  if X_train.shape[0] != Y_train.size:
    X_train = np.array(X_train).T
  if X_test.shape[0] != Y_test.size:
    X_test = X_test.T

  tau = estim_tau(X_train)
  print(tau)
  thresholds = compute_thresholds(tau, 'rff')
  s_minus = np.min(thresholds)
  s_plus = np.max(thresholds)
  print(s_minus, s_plus)
  # # # DEBUG
  # s = np.arange(X_train.shape[0] )
  # np.random.seed(0)
  # np.random.shuffle(s)
  # X_train = X_train[s, :]
  # Y_train = Y_train[s]
  # X_train, Y_train, X_test, Y_test = \
  # X_train[:int(s.size * 1 / 5), :], Y_train[:int(s.size * 1 / 5)], X_test[:int(s.size * 1 / 5), :], Y_test[:int(s.size * 1 / 5)]
  # print("test ", X_train.shape, Y_train.shape)
  assert X_train.shape[0] == Y_train.shape[0]
  assert X_test.shape[0] == Y_test.shape[0]
  assert X_train.shape[0] != X_test.shape[0]
  return X_train, X_test, Y_train, Y_test, s_minus, s_plus
