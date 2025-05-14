from utils import *
from imports import *

people = np.zeros(100)
cluster_time = 100
timepoint = 1198
sw = np.load('real_results/sw_results.npz')
swb = np.load('real_results/inter_data/3length_sdae200_sw.npz')['swb']
con = np.load('real_results/con_results.npz')
conb = np.load('real_results/inter_data/3length_sdae200_con.npz')['conb']
cony = con['dfc']
swy = sw['dfc']
sw_count = np.zeros((cluster_time,people.shape[0]))
for clu in range(cluster_time):
    for j in range(people.shape[0]):
        index = swb[clu,timepoint * j:timepoint * (j + 1)]
        count = count_transitions(index)
        sw_count[clu,j] = count
    print(ttest_ind(sw_count[clu,:46],sw_count[clu,46:]).pvalue)
