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
cony = np.load('conyneed.npy')
con_flu = np.zeros((people.shape[0],2))
sw_flu = np.zeros((people.shape[0],2))
for i in range(people.shape[0]):
    amplitude = np.mean(np.abs(cony[i]),axis=0)
    variability = np.std(cony[i],axis=0)
    amplitude = np.mean(amplitude)
    variability = np.mean(variability)
    con_flu[i,0] = amplitude
    con_flu[i,1] = variability


for i in range(people.shape[0]):
    amplitude = np.mean(np.abs(swy[i]),axis=0)
    variability = np.std(swy[i],axis=0)
    amplitude = np.mean(amplitude)
    variability = np.mean(variability)
    sw_flu[i,0] = amplitude
    sw_flu[i,1] = variability
pass
