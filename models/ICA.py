from sklearn.decomposition import FastICA
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from utils import *

start_time = time.time()

n_dim = 8
# mat_data = scipy.io.loadmat('/longlab/duan/pycharm_programs/formal_project/simData_60ROI_800TR_sub1.mat')
data = np.load(paras.all_tc)[:200]
data = data + 0.8 * np.random.random(data.shape)
index = np.load(paras.all_index)[:200]
labels = np.zeros_like(index)
if labels.shape.__len__() == 1:
    labels = labels[None,:]
# data = mat_data['TC']
if data.shape.__len__() == 2:
    data = data[None, :]
n_sub, n_roi, n_tr = data.shape
data = data.transpose(0, 2, 1)
# data = data.reshape(-1, n_roi)
for j in range(n_sub):
# 初始化 FastICA
    ica = FastICA(n_components=n_dim, random_state=0,whiten='unit-variance')

    # 应用 FastICA 到数据
    S = ica.fit_transform(data[j])  # S 是分离出的成分
    A = ica.mixing_  # A 是混合矩阵

    # 初始化一个空列表来存储重构的数据
    reconstructed_sources = []

    # 对于每个独立成分，使用它和混合矩阵重构原始数据
    for i in range(S.shape[1]):
        S_temp = np.zeros_like(S)
        S_temp[:, i] = S[:, i]
        reconstructed_source = np.dot(S_temp, A.T)
        reconstructed_sources.append(reconstructed_source)

    need = np.zeros((n_tr,n_roi,n_dim))
    for i in range(n_dim):
        need[:,:,i] = reconstructed_sources[i]
    centroids,label = cluster(corshow(need,show=False),4)
    labels[j] = label
    # label = label.reshape(n_sub,-1)
    # index = index.reshape(n_sub,-1)
    print(j)
if j > 1:
    tmp=np.zeros(200)
    for i in range(200):
        # print('ARI = {}'.format(metrics.adjusted_rand_score(labels[i], index[i])))
        tmp[i] = metrics.adjusted_rand_score(labels[i], index[i])
else:
    print('ARI = {}'.format(metrics.adjusted_rand_score(labels.squeeze(), index)))
states_show(centroids)
# 现在，reconstructed_sources 包含 8 个分别由每个独立成分重构的 90x1200 数据
pass