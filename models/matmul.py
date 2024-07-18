import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from paras import paras
from sklearn.decomposition import PCA
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from utils import *
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
# start_time = time.time()
from scipy.stats import skew, kurtosis

# if __name__ == '__main__':
#     all_data = np.load(paras.all_tc)[0]
#     if all_data.shape.__len__() == 2:
#         all_data = all_data[None,:]
#     n_sub,n_roi,n_tr = all_data.shape
#     all_data = all_data.transpose(0,2,1)
#     all_data = all_data.reshape(-1,n_roi)
#     need = np.zeros((all_data.shape[0],all_data.shape[1],all_data.shape[1]))
#     for i in range(all_data.shape[0]):
#         need[i] = np.matmul(all_data[i][:,None],all_data[i][:,None].T)
#     # all_data_expanded = all_data[:, :, None]  # 扩展维度以匹配形状
#     # need = np.matmul(all_data_expanded, all_data_expanded.transpose(0,2,1))
#     states,index = cluster(need,5)
#     states_show(states)
#     end_time = time.time()
#     print(end_time-start_time)
#     pass


def mm(all_data):
    if all_data.shape.__len__() == 2:
        all_data = all_data[None,:]
    n_sub,n_roi,n_tr = all_data.shape
    all_data = all_data.transpose(0,2,1)
    # all_data = all_data.reshape(-1,n_roi)
    need = np.zeros((n_sub,n_tr,n_roi,n_roi))
    for i in range(n_sub):
        for j in range(n_tr):
            need[i,j] = np.matmul(all_data[i,j][:,None],all_data[i,j][:,None].T)
            need[i,j] = ex(need[i,j])
    return need
import numpy as np

def mm_optimized(all_data):
    if all_data.ndim == 2:
        all_data = all_data[None, :]
    all_data = all_data.transpose(0, 2, 1)
    # Compute outer product for all subjects and time points simultaneously
    need = np.einsum('sij, sik -> sijk', all_data, all_data)
    # Apply the function 'ex' across all matrices
    # need = np.vectorize(ex, signature='(n,m)->(n,m)')(need)
    return need


if __name__ == '__main__':
    t=time.time()
    a=mm_optimized(np.random.randn(800,90,1200))
    print(a.shape)
    print(time.time()-t)
    # pass