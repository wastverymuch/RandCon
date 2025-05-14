import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import keyword
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
# 下面是cluster.py里的import
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import pearsonr
import itertools
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import faiss
import time
from models.conv import conv
from models.mtd import mtd
from models.sw import sliding_windows
from hmmlearn import hmm
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import hilbert
import pandas as pd
from models.phase import phase
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error as MSE
import torch
from datetime import datetime
# 下面是cluster.py里的import
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from line_profiler import LineProfiler
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import gc