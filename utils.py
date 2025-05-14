from sklearn.metrics import mean_squared_error as MSE
import faiss
import time
import torch
# 下面是cluster.py里的import
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import shapiro, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

def states_show(x,dpi=400, save=False, title=None,   show=True,save_path = '.'):
    num_plots = x.shape[0]
    if x.shape.__len__() == 2:
        a = int(np.sqrt(x.shape[1]))
    else:
        a = x.shape[-1]
    x = x.reshape(-1, a, a)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))
    plt.figure(dpi=dpi)
    vmin, vmax = -1, 1
    for i in range(num_plots):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x[i], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=-1, vmax=1)), ax=plt.gcf().axes,
                     orientation='vertical', fraction=0.046, pad=0.04)

    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle(' ')

    if save:
        plt.savefig(save_path,dpi=dpi)
    if show == True:
        plt.show()

def davies_compare_old_(centroids,indexes,t_clusters,model_tc):
    result = []
    for clu in range(t_clusters):
        result.append(davies_bouldin_score(model_tc.reshape(-1, 8100), indexes[clu]))
    return result.index(min(result))


def davies_compare_old(centroids, indexes, t_clusters, model_tc):
    reshaped_model_tc = model_tc.reshape(-1, 10)
    results = Parallel(n_jobs=-1)(
        delayed(davies_bouldin_score)(reshaped_model_tc, indexes[clu]) for clu in range(t_clusters))

    return results.index(min(results)), np.stack(results)

def davies_compare(centroids, indexes, t_clusters, model_tc, chunk_size=24000):
    def process_chunk(start, end):
        reshaped_model_tc = model_tc[start:end].reshape(-1, model_tc.shape[-1] ** 2)
        return [davies_bouldin_score(reshaped_model_tc, indexes[clu]) for clu in range(t_clusters)]

    n_samples = model_tc.shape[0]
    results = []

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_results = process_chunk(start, end)
        results.extend(chunk_results)

    return results.index(min(results)), np.stack(results)

def compare(x,y,a,b):
    ari = metrics.adjusted_rand_score(y,b.reshape(-1))
    a=a.reshape(a.shape[0],-1)
    centroids = x
    mse, cosine_sim,eud,mand,cor,prd = align_and_calculate_metrics(centroids, a)
    kl = KL(b,y,a.shape[0])
    overlap_ratio = overlap(b,y,a.shape[0])
    return ari,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd
def compare_diff_time(x,y,a,b,t=100):
    kl = KL(b,y,a.shape[0])
    a=a.reshape(a.shape[0],-1)
    centroids = x
    mse, cosine_sim,eud,mand,cor,prd = align_and_calculate_metrics(centroids, a)
    return kl,mse, cosine_sim,eud,mand,cor,prd

def KL(index, label,states):
    index = index.reshape(-1)
    sj = cal_sj(states, index)
    ysj = cal_sj(states,label)
    KLs = np.zeros((states,states))
    for key in sj:
        for KEY in ysj:
            newsj,newysj = match_length_with_zeros(sj[key],ysj[KEY])
            newsj = np.nan_to_num(newsj,1e-10)
            newsj = np.where(newsj == 0 , 1e-10 , newsj)
            newysj = np.where(newysj == 0, 1e-10, newysj)
            KLs[key,KEY] = entropy(newsj,newysj)
    row_ind, col_ind = linear_sum_assignment(KLs)
    return KLs[row_ind, col_ind].mean()

def overlap(index, label,states):
    index = index.reshape(-1)
    ov =calculate_overlap_probability(index,label,states = states)
    row_ind, col_ind = linear_sum_assignment(-ov)
    return ov[row_ind,col_ind].sum()
def calculate_overlap_probability(ts1, ts2,states):
    overlap_matrix = np.zeros((states, states))
    ts1 = ts1.astype(int)
    ts2 = ts2.astype(int)
    for i in range(len(ts1)):
        overlap_matrix[ts1[i], ts2[i]] += 1
    overlap_probability = overlap_matrix / len(ts1)
    return overlap_probability
def match_length_with_zeros(list1, list2):
    len1, len2 = len(list1), len(list2)
    if len1 > len2:
        list2 = np.pad(list2, (0, len1 - len2), 'constant')
    elif len2 > len1:
        list1 = np.pad(list1, (0, len2 - len1), 'constant')
    return np.array(list1), np.array(list2)
def cal_sj(K, idx):
    subT = len(idx)
    sojournTimes = {k: [] for k in range(K)}
    sub = len(idx) // subT
    idx = np.array(idx)

    for i in range(sub):
        idx1 = idx[subT * i:subT * (i + 1)]
        currentK = idx1[0]
        sojourn = 1

        for t in range(1, len(idx1)):
            if idx1[t] == currentK:
                sojourn += 1
            else:
                sojournTimes[currentK].append(sojourn)
                currentK = idx1[t]
                sojourn = 1
        sojournTimes[currentK].append(sojourn)

    return sojournTimes

def align_and_calculate_metrics(vector1, vector2):
    correspondences = []
    for i in range(vector1.shape[0]):
        nearest_index = distance.cdist([vector1[i]], vector2).argmin()
        correspondences.append((i, nearest_index))
    aligned_vector1 = np.array([vector1[i] for i, _ in correspondences])
    aligned_vector2 = np.array([vector2[j] for _, j in correspondences])
    mse = mean_squared_error(aligned_vector1, aligned_vector2)
    mse_per_row = []
    for i in range(aligned_vector1.shape[0]):
        mse_per_row.append(mean_squared_error(aligned_vector1[i], aligned_vector2[i]))
    mse = np.mean(mse_per_row)
    cosine_sim = cosine_similarity(aligned_vector1, aligned_vector2)
    cosine_sim = np.diag(cosine_sim).mean()
    eud , mand , cor , prd = calculate_metrics(vector1, vector2)
    return mse, cosine_sim ,eud,mand,cor,prd


def calculate_metrics(matrix1, matrix2):

    euclidean_distances = []
    manhattan_distances = []
    correlation_coefficients = []
    prds = []

    for i in range(matrix1.shape[0]):
        eu_dist = euclidean(matrix1[i], matrix2[i])
        euclidean_distances.append(eu_dist)
        man_dist = cityblock(matrix1[i], matrix2[i])
        manhattan_distances.append(man_dist)
        corr_coef, _ = pearsonr(matrix1[i], matrix2[i])
        correlation_coefficients.append(corr_coef)
        prd = np.sqrt(np.sum((matrix1[i] - matrix2[i]) ** 2) / np.sum(matrix1[i] ** 2)) * 100
        prds.append(prd)
    return np.mean(euclidean_distances), np.mean(manhattan_distances), np.mean(correlation_coefficients), np.mean(prds)

def faisscluster_old(x,t_clusters,m=4,):
    if x.shape[-1] == x.shape[-2]:
        x = x.reshape(-1,x.shape[-1] ** 2)
    centroids = []
    indexs = []
    for i in range(t_clusters):
        kmeans = faiss.Kmeans(x.shape[-1], m, niter=20, verbose=False, gpu=False,seed = i + int(time.time()))
        kmeans.train(x)
        centroids.append(kmeans.centroids)
        _,a=kmeans.index.search(x,1)
        indexs.append(a.squeeze())
    return centroids,indexs

def kmeans_task(x, m, seed):
    kmeans = faiss.Kmeans(x.shape[-1], m, niter=20, verbose=False, gpu=False, seed=seed)
    kmeans.train(x)
    _, a = kmeans.index.search(x, 1)
    return kmeans.centroids, a.squeeze()


def convert_to_lower_triangular_no_diagonal(input_matrix):

    if input_matrix.__len__() == 1:
        input_matrix = input_matrix[None,:]
    num_matrices, original_dim = input_matrix.shape
    matrix_size = int(np.sqrt(original_dim))

    tril_indices = np.tril_indices(matrix_size, k=-1)
    num_lower_triangular_elements = len(tril_indices[0])
    output_matrix = np.zeros((num_matrices, num_lower_triangular_elements))

    for i in range(num_matrices):
        full_matrix = input_matrix[i].reshape(matrix_size, matrix_size)
        lower_triangular_no_diag = full_matrix[tril_indices]
        output_matrix[i] = lower_triangular_no_diag

    return output_matrix


def reconstruct_full_matrices_from_lower_triangular(input_matrix):

    num_matrices, lower_dim = input_matrix.shape
    matrix_size = int((1 + np.sqrt(1 + 8 * lower_dim)) // 2)
    tril_indices = np.tril_indices(matrix_size, k=-1)
    output_matrix = np.zeros((num_matrices, matrix_size * matrix_size))

    for i in range(num_matrices):
        full_matrix = np.eye(matrix_size)
        full_matrix[tril_indices] = input_matrix[i]
        full_matrix = full_matrix + full_matrix.T - np.diag(full_matrix.diagonal())
        output_matrix[i] = full_matrix.flatten()

    return output_matrix

def faisscluster(x, t_clusters, m=4,convert = True):
    if x.shape[-1] == x.shape[-2]:
        x = x.reshape(-1, x.shape[-1] ** 2)
    if convert:
        x = convert_to_lower_triangular_no_diagonal(x)
    centroids = []
    indexs = []
    current_time = int(time.time())

    seeds = [i + current_time for i in range(t_clusters)]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda seed: kmeans_task(x, m, seed), seeds))

    for cent, ind in results:
        centroids.append(cent)
        indexs.append(ind)

    return centroids, indexs

def corshow(x):
    if isinstance(x,torch.Tensor):
        x=x.cpu().detach()
    t,r,d = x.shape
    y=np.zeros((t,r,r))
    for i in range(t):
        y[i] = np.corrcoef(x[i])
    return y

def standardize_vector(vector):
    mean = np.mean(vector)
    std_dev = np.std(vector)
    standardized_vector = (vector - mean) / std_dev
    return standardized_vector

def scatter(a):
    length = a.shape[0]
    plt.scatter(np.arange(length),a)
    plt.show()
def compare_index(a,b,t1,t2,title = None):
    plt.plot(a[t1:t2])
    plt.scatter(np.arange(t2-t1),b[t1:t2],color = 'red')
    if title == None:
        plt.title(f'{t1} to {t2} TR')
    else:
        plt.title(title)
    plt.show()

def fisher_r_to_z(r):
    if r == 1:
        r = 0.999999
    return 0.5 * np.log((1 + r) / (1 - r))
def fisher_r_to_z_array(r_array):
    r_array = np.clip(r_array, -0.999999, 0.999999)  # Clip the values to handle r == 1 or r == -1
    return 0.5 * np.log((1 + r_array) / (1 - r_array))

def fisher_rtoz(matrix):

    matrix = np.asarray(matrix, dtype=np.float64)
    z_matrix = np.empty_like(matrix, dtype=np.float64)
    z_matrix[matrix == 1] = np.inf
    z_matrix[matrix == -1] = -np.inf
    z_matrix[matrix == 0] = 0
    mask = (matrix != 1) & (matrix != -1) & (matrix != 0)
    z_matrix[mask] = 0.5 * np.log((1 + matrix[mask]) / (1 - matrix[mask]))

    return z_matrix


def count_transitions(sequence):
    transitions = 0
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            transitions += 1

    return transitions
def mean_dwell_time(sequence):
    durations = {0: [], 1: [], 2: [], 3: []}
    n = len(sequence)
    current_digit = sequence[0]
    start_index = 0

    for i in range(1, n):
        if sequence[i] != current_digit:
            duration = i - start_index
            durations[current_digit].append(duration)
            current_digit = sequence[i]
            start_index = i

    duration = n - start_index
    durations[current_digit].append(duration)
    average_durations = np.zeros(4)
    for digit, duration_list in durations.items():
        if duration_list:
            average_durations[digit] = sum(duration_list) / len(duration_list)

    return average_durations
