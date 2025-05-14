from utils import *
from imports import *

people = np.zeros(100)
cluster_time = 100
timepoint = 1198
sw = np.load('real_results/sw_results.npz')
swb = np.load('real_results/inter_data/3length_sdae200_sw.npz')['swb']
con = np.load('real_results/con_results.npz')
conb = np.load('real_results/inter_data/3length_sdae200_con.npz')['conb']
swy = sw['dfc']
con_cosines = np.zeros((100,1198))
cony = np.load('real_results/inter_data/cony.npy')
for i in range(100):
    con_mean = cony[i].mean(0).reshape(-1)[None,:]
    con_mean = fisher_r_to_z_array(con_mean)
    for time in range(1198):
        con_cosines[i,time] = cosine_similarity(fisher_r_to_z_array(cony[i,time].reshape(-1)[None,:]),con_mean)[0,0]
concos = con_cosines.mean(-1)
sw_cosines = np.zeros((100,1198))
for i in range(100):
    sw_mean = swy[i].mean(0).reshape(-1)[None,:]
    sw_mean = fisher_r_to_z_array(sw_mean)
    for time in range(1198):
        sw_cosines[i,time] = cosine_similarity(fisher_r_to_z_array(swy[i,time].reshape(-1)[None,:]),sw_mean)[0,0]
swcos = sw_cosines.mean(-1)
print(ttest_ind(swcos[:46],swcos[46:]).pvalue)
print(ttest_ind(concos[:46],concos[46:]).pvalue)

pass