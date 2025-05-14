import numpy as np

from utils import *
from imports import *
real_data = np.load('real_data/subs100_90_1200.npz')
male = [1, 2, 3, 5, 9, 11, 12, 16, 17, 19, 22, 23, 25, 26, 27, 33, 36, 37, 42, 45, 46, 50, 54, 55, 56, 58, 61, 62, 63,
        64, 67, 69, 71, 75, 76, 77, 79, 80, 82, 88, 89, 90, 93, 94, 95, 98]
female = [0, 4, 6, 7, 8, 10, 13, 14, 15, 18, 20, 21, 24, 28, 29, 30, 31, 32, 34, 35, 38, 39, 40, 41, 43, 44, 47, 48, 49,
          51, 52, 53, 57, 59, 60, 65, 66, 68, 70, 72, 73, 74, 78, 81, 83, 84, 85, 86, 87, 91, 92, 96, 97, 99]
all_data = real_data['data']
maledata = all_data[male]
femaledata = all_data[female]
people = np.vstack((maledata,femaledata))
for i in range(people.shape[0]):
    for j in range(people.shape[1]):
        people[i,j] = standardize_vector(people[i,j])
window_length = 3
timepoint = people.shape[-1] - window_length + 1
swy = np.zeros((people.shape[0],timepoint,people.shape[1],people.shape[1]))
con_time = 40
cony = np.zeros((people.shape[0],timepoint,people.shape[1],people.shape[1]))
for j in range(con_time):
    for i in range(people.shape[0]):
        cony[i] = conv(people[i],  out_channels=2048, windows_length=window_length,padding = False)
    np.save(f'real_results/inter_data/conys/con{j}.npy',cony)
cluster_time = 100
swa,swb = faisscluster(swy,cluster_time,convert=0)
_, swdavies = davies_compare_old(swa, swb, cluster_time, swy)
cona,conb = faisscluster(cony,cluster_time,convert=0)
_, condavies = davies_compare_old(cona,conb,cluster_time, cony)
cluster_time = 100
swy = swy.reshape(-1, 8100)
cony = cony.reshape(-1, 8100)

for i in range(cluster_time):
    score = davies_bouldin_score(cony, conb[i])
    states_show(reconstruct_full_matrices_from_lower_triangular(cona[i]), save=True, show=False,
                save_path=f'real_results/inter_data/3length_no04_con/{i}.jpg',
                title=f'{score}')
for i in range(cluster_time):
    score = davies_bouldin_score(swy,swb[i])
    states_show(reconstruct_full_matrices_from_lower_triangular(swa[i]), save=True, show=False,
                save_path=f'real_results/inter_data/3length_no04_sw/{i}.jpg',title = f'{score}')


people = np.zeros(100)
cluster_time = 100
timepoint = 1198
sw = np.load('real_results/sw_results.npz')
swb = np.load('real_results/inter_data/3length_sdae200_sw.npz')['swb']
con = np.load('real_results/con_results.npz')
conb = np.load('real_results/inter_data/3length_sdae200_con.npz')['conb']
sw_occp = np.zeros((cluster_time,4))
sw_occ = np.zeros((cluster_time,people.shape[0],4))
for a in range(cluster_time):
    manind = swb[a][:timepoint * 46]
    womanind = swb[a][timepoint * 46:]
    manratio = np.zeros((46,4))
    for i in range(46):
        for j in range(4):
            index = manind[timepoint * i:timepoint * (i+1)]
            manratio[i, j] = np.sum(index == j) / timepoint

    womanratio = np.zeros((54, 4))
    for i in range(54):
        for j in range(4):
            index = womanind[timepoint * i:timepoint * (i+1)]
            womanratio[i, j] = np.sum(index == j) / timepoint
    sw_occ[a] = np.vstack((manratio,womanratio))
    for i in range(4):
        sw_occp[a,i] = ttest_ind(manratio[:, i], womanratio[:, i]).pvalue


con_occp = np.zeros((cluster_time, 4))
con_occ = np.zeros((cluster_time,people.shape[0],4))
for a in range(cluster_time):
    manind = conb[a][:timepoint * 46]
    womanind = conb[a][timepoint * 46:]
    manratio = np.zeros((46, 4))
    for i in range(46):
        for j in range(4):
            index = manind[timepoint * i:timepoint * (i + 1)]
            manratio[i, j] = np.sum(index == j) / timepoint
    womanratio = np.zeros((54, 4))
    for i in range(54):
        for j in range(4):
            index = womanind[timepoint * i:timepoint * (i + 1)]
            womanratio[i, j] = np.sum(index == j) / timepoint
    con_occ[a] = np.vstack((manratio, womanratio))
    for i in range(4):
        con_occp[a, i] = ttest_ind(manratio[:, i], womanratio[:, i]).pvalue

condt = np.zeros((cluster_time,people.shape[0],4))
condtp = np.zeros((cluster_time,4))
for a in range(cluster_time):
    for i in range(people.shape[0]):
        index = conb[a][timepoint * i:timepoint * (i + 1)]
        condt[a,i] = mean_dwell_time(index)
    for j in range(4):
        condtp[a,j] = ttest_ind(condt[a,:46,j],condt[a,46:,j]).pvalue

swdt = np.zeros((cluster_time,people.shape[0],4))
swdtp = np.zeros((cluster_time,4))
for a in range(cluster_time):
    for i in range(people.shape[0]):
        index = swb[a][timepoint * i:timepoint * (i + 1)]
        swdt[a,i] = mean_dwell_time(index)
    for j in range(4):
        swdtp[a,j] = ttest_ind(swdt[a,:46,j],swdt[a,46:,j]).pvalue

pass












# 实验debug记录
# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# /longlab/duan/conda_envs/start/bin/python3.11 /longlab/duan/.pycharm_helpers/pydev/pydevd.py --multiprocess --qt-support=auto --client localhost --port 32825 --file /longlab/duan/pycharm_programs/formal_project/exp_real_data_time.py
# Connected to pydev debugger (build 232.10072.31)
# np.sum(sw_occp<0.05)
# Python 3.11.4 (main, Jul  5 2023, 13:45:01) [GCC 11.2.0]
# Type 'copyright', 'credits' or 'license' for more information
# IPython 8.17.2 -- An enhanced Interactive Python. Type '?' for help.
# PyDev console: using IPython 8.17.2
# Out[1]: 0
# np.sum(con_occp<0.05)
# Out[2]: 1
# np.where(con_occp<0.05)
# Out[3]: (array([95]), array([0]))
# con_occ[95]
# Out[4]:
# array([[0.04590985, 0.24791319, 0.24624374, 0.45993322],
#        [0.16277129, 0.16611018, 0.4457429 , 0.22537563],
#        [0.15108514, 0.28714524, 0.40734558, 0.15442404],
#        [0.14858097, 0.30133556, 0.50417362, 0.04590985],
#        [0.21869783, 0.15525876, 0.57429048, 0.05175292],
#        [0.07762938, 0.34307179, 0.23288815, 0.34641068],
#        [0.18196995, 0.28547579, 0.34056761, 0.19198664],
#        [0.0918197 , 0.20784641, 0.42904841, 0.27128548],
#        [0.11018364, 0.2754591 , 0.30717863, 0.30717863],
#        [0.05592654, 0.2721202 , 0.31385643, 0.35809683],
#        [0.27796327, 0.15025042, 0.55342237, 0.01836394],
#        [0.19782972, 0.24791319, 0.47579299, 0.07846411],
#        [0.1360601 , 0.19616027, 0.45075125, 0.21702838],
#        [0.24624374, 0.19198664, 0.48664441, 0.07512521],
#        [0.21202003, 0.10350584, 0.67863105, 0.00584307],
#        [0.29382304, 0.10267112, 0.58096828, 0.02253756],
#        [0.21702838, 0.16360601, 0.55592654, 0.06343907],
#        [0.22287145, 0.21452421, 0.44824708, 0.11435726],
#        [0.0409015 , 0.2345576 , 0.24123539, 0.48330551],
#        [0.1360601 , 0.23622705, 0.36143573, 0.26627713],
#        [0.15275459, 0.22203673, 0.38480801, 0.24040067],
#        [0.03088481, 0.23622705, 0.28046745, 0.4524207 ],
#        [0.15025042, 0.139399  , 0.58764608, 0.12270451],
#        [0.10851419, 0.32136895, 0.41235392, 0.15776294],
#        [0.12353923, 0.23873122, 0.45826377, 0.17946578],
#        [0.13689482, 0.18280467, 0.5851419 , 0.0951586 ],
#        [0.14190317, 0.26794658, 0.39816361, 0.19198664],
#        [0.10267112, 0.24040067, 0.39732888, 0.25959933],
#        [0.12771285, 0.23706177, 0.53923205, 0.09599332],
#        [0.1293823 , 0.38898164, 0.32387312, 0.15776294],
#        [0.16110184, 0.34056761, 0.29716194, 0.20116861],
#        [0.14774624, 0.20534224, 0.52337229, 0.12353923],
#        [0.10100167, 0.29966611, 0.37562604, 0.22370618],
#        [0.15191987, 0.19365609, 0.54674457, 0.10767947],
#        [0.16110184, 0.29298831, 0.36143573, 0.18447412],
#        [0.11352254, 0.30050083, 0.42904841, 0.15692821],
#        [0.12520868, 0.21953255, 0.54590985, 0.10934891],
#        [0.03422371, 0.25459098, 0.3130217 , 0.39816361],
#        [0.04507513, 0.34557596, 0.22287145, 0.38647746],
#        [0.10934891, 0.21702838, 0.47662771, 0.19699499],
#        [0.14774624, 0.16193656, 0.64023372, 0.05008347],
#        [0.06343907, 0.26293823, 0.33555927, 0.33806344],
#        [0.15859766, 0.2754591 , 0.38814691, 0.17779633],
#        [0.06844741, 0.35141903, 0.25292154, 0.32721202],
#        [0.20283806, 0.20200334, 0.46661102, 0.12854758],
#        [0.221202  , 0.20784641, 0.48163606, 0.08931553],
#        [0.20534224, 0.1836394 , 0.55342237, 0.05759599],
#        [0.1327212 , 0.26711185, 0.38564274, 0.21452421],
#        [0.2721202 , 0.16193656, 0.53589316, 0.03005008],
#        [0.10016694, 0.30300501, 0.31051753, 0.28631052],
#        [0.20116861, 0.12103506, 0.63522538, 0.04257095],
#        [0.11853088, 0.29883139, 0.42988314, 0.15275459],
#        [0.16110184, 0.25041736, 0.360601  , 0.2278798 ],
#        [0.25208681, 0.09348915, 0.61435726, 0.04006678],
#        [0.17863105, 0.26961603, 0.36978297, 0.18196995],
#        [0.3196995 , 0.1803005 , 0.43656093, 0.06343907],
#        [0.12270451, 0.27712855, 0.38898164, 0.21118531],
#        [0.28130217, 0.11018364, 0.54841402, 0.06010017],
#        [0.27712855, 0.1903172 , 0.50751252, 0.02504174],
#        [0.21368948, 0.21953255, 0.37896494, 0.18781302],
#        [0.3230384 , 0.17863105, 0.38313856, 0.11519199],
#        [0.20450751, 0.18864775, 0.54340568, 0.06343907],
#        [0.13689482, 0.28046745, 0.42320534, 0.15943239],
#        [0.13856427, 0.31469115, 0.28631052, 0.26043406],
#        [0.29966611, 0.17529215, 0.50918197, 0.01585977],
#        [0.06594324, 0.2245409 , 0.30383973, 0.40567613],
#        [0.139399  , 0.22621035, 0.46744574, 0.16694491],
#        [0.24874791, 0.29799666, 0.38731219, 0.06594324],
#        [0.16026711, 0.31552588, 0.33055092, 0.19365609],
#        [0.04006678, 0.21452421, 0.34140234, 0.40400668],
#        [0.15025042, 0.21702838, 0.40651085, 0.22621035],
#        [0.12353923, 0.26210351, 0.29465776, 0.3196995 ],
#        [0.20200334, 0.20450751, 0.43071786, 0.16277129],
#        [0.0984975 , 0.22287145, 0.38647746, 0.29215359],
#        [0.16110184, 0.24123539, 0.52337229, 0.07429048],
#        [0.24040067, 0.15191987, 0.59599332, 0.01168614],
#        [0.16277129, 0.08430718, 0.74624374, 0.0066778 ],
#        [0.19198664, 0.23372287, 0.37813022, 0.19616027],
#        [0.12270451, 0.31719533, 0.38397329, 0.17612688],
#        [0.25041736, 0.11018364, 0.60517529, 0.03422371],
#        [0.20450751, 0.26126878, 0.30801336, 0.22621035],
#        [0.07512521, 0.22871452, 0.48664441, 0.20951586],
#        [0.1803005 , 0.24540902, 0.35392321, 0.22036728],
#        [0.18196995, 0.27378965, 0.29966611, 0.24457429],
#        [0.04173623, 0.29382304, 0.33472454, 0.32971619],
#        [0.08347245, 0.2278798 , 0.34891486, 0.33973289],
#        [0.05342237, 0.24958264, 0.30550918, 0.39148581],
#        [0.09015025, 0.27128548, 0.43739566, 0.20116861],
#        [0.22203673, 0.27712855, 0.33472454, 0.16611018],
#        [0.21786311, 0.17946578, 0.55342237, 0.04924875],
#        [0.09933222, 0.34808013, 0.33889816, 0.21368948],
#        [0.12520868, 0.09682805, 0.76377295, 0.01419032],
#        [0.25959933, 0.09265442, 0.6327212 , 0.01502504],
#        [0.32721202, 0.07679466, 0.58263773, 0.01335559],
#        [0.08263773, 0.29716194, 0.4148581 , 0.20534224],
#        [0.09432387, 0.32470785, 0.35559265, 0.22537563],
#        [0.20367279, 0.18280467, 0.5033389 , 0.11018364],
#        [0.19616027, 0.21619366, 0.48664441, 0.10100167],
#        [0.21702838, 0.14357262, 0.56594324, 0.07345576],
#        [0.11769616, 0.21285476, 0.48831386, 0.18113523]])
# conx = con_occ[95]
# np.where(sw_occ==sw_occ.min())
# Out[6]:
# (array([ 2, 39, 44, 49, 51, 51, 56, 59, 59, 61, 69, 75, 88, 88, 92, 96, 98]),
#  array([76, 76, 76, 76, 76, 93, 76, 76, 93, 76, 76, 76, 76, 93, 76, 76, 76]),
#  array([0, 1, 0, 2, 2, 2, 1, 3, 3, 3, 2, 1, 1, 1, 2, 3, 2]))
# np.where(sw_occp==sw_occp.min())
# Out[7]: (array([95]), array([0]))
# swx = sw_occ[95]
# sw_occp[95]
# Out[9]: array([0.10156081, 0.48293513, 0.26659656, 0.15771139])
# np.save('occ_result.npz',sw=swx,con=conx)
# Traceback (most recent call last):
#   File "/longlab/duan/conda_envs/start/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3548, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-10-3892b8386c72>", line 1, in <module>
#     np.save('occ_result.npz',sw=swx,con=conx)
# TypeError: save() got an unexpected keyword argument 'sw'
# np.savez('occ_result.npz',sw=swx,con=conx)
# np.where(condtp<0.05)
# Out[12]: (array([12, 42, 58, 67, 98]), array([1, 3, 0, 0, 0]))
# condtp[12]
# Out[13]: array([0.40466848, 0.04891415, 0.49095038, 0.07013914])
# condtp[42]
# Out[14]: array([0.46828168, 0.36023956, 0.1606918 , 0.04511784])
# condtp[58]
# Out[15]: array([0.00844227, 0.29614849, 0.10224256, 0.65222716])
# condtp[67]
# Out[16]: array([0.04768391, 0.52112631, 0.38507595, 0.1409225 ])
# condtp[98]
# Out[17]: array([0.04771988, 0.42030994, 0.32877375, 0.20228212])
# swdtp[58]
# Out[18]: array([0.06122532, 0.15821659, 0.04062541, 0.16433689])
# np.where(swdtp==swdtp.min())
# Out[19]: (array([99]), array([3]))
# swdtp[99]
# Out[20]: array([0.13934347, 0.06139893, 0.14079381, 0.03139924])
# np.savez('dt_result.npz',sw=swdt[99],con=condtp[58])
# np.savetxt('occ_sw.csv',swx.T,delimiter=',')
# conx.mean(0)
# Out[23]: array([0.15813022, 0.22877295, 0.43737062, 0.17572621])
# swx.mean(0)
# Out[24]: array([0.41856427, 0.28254591, 0.2030384 , 0.09585142])
# tmp = np.load('/longlab/duan/pycharm_programs/formal_project/real_results/inter_data/3length_sw.npz')['swy']
# tmp1 = tmp.reshape(-1,8100)[np.where(swb[95] == 0)]
# states_show(tmp1.mean(0))
# Traceback (most recent call last):
#   File "/longlab/duan/conda_envs/start/lib/python3.11/site-packages/IPython/core/interactiveshell.py", line 3548, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-27-0f69ad2bada9>", line 1, in <module>
#     states_show(tmp1.mean(0))
#   File "/longlab/duan/pycharm_programs/formal_project/utils.py", line 39, in states_show
#     x = x.reshape(-1, a, a)
#         ^^^^^^^^^^^^^^^^^^^
# ValueError: cannot reshape array of size 8100 into shape (8100,8100)
# states_show(tmp1.mean(0)[None,:])
# tmp2 = np.load('/longlab/duan/pycharm_programs/formal_project/real_results/inter_data/3length_con.npz')['cony']
# tmp3 = tmp2.reshape(-1,8100)[np.where(conb[95] == 2)]
# states_show(tmp3.mean(0)[None,:])
# tmp3 = tmp2.reshape(-1,8100)[np.where(conb[95] == 0)]
# states_show(tmp3.mean(0)[None,:])
# tmpstate=[]
# tmpstate=np.zeros((4,8100))
# tmpstate[0] = tmp2.reshape(-1,8100)[np.where(conb[95] == 0)].mean(0)
# tmpstate[1] = tmp2.reshape(-1,8100)[np.where(conb[95] == 1)].mean(0)
# tmpstate[2] = tmp2.reshape(-1,8100)[np.where(conb[95] == 2)].mean(0)
# tmpstate[3] = tmp2.reshape(-1,8100)[np.where(conb[95] == 3)].mean(0)
# states_show(tmpstate)
# np.savetxt('dt_sw.csv',swdt[99].T,delimiter=',')
# np.savetxt('dt_con.csv',condtp[58].T,delimiter=',')
# np.savetxt('dt_con.csv',condt[58].T,delimiter=',')
# tmpstate[0] = tmp2.reshape(-1,8100)[np.where(conb[58] == 0)].mean(0)
# tmpstate[1] = tmp2.reshape(-1,8100)[np.where(conb[58] == 1)].mean(0)
# tmpstate[2] = tmp2.reshape(-1,8100)[np.where(conb[58] == 2)].mean(0)
# tmpstate[3] = tmp2.reshape(-1,8100)[np.where(conb[58] == 3)].mean(0)
# states_show(tmpstate)
# condtp[58]
# Out[46]: array([0.00844227, 0.29614849, 0.10224256, 0.65222716])
# swdtp[99]
# Out[47]: array([0.13934347, 0.06139893, 0.14079381, 0.03139924])
# tmpstate[0] = tmp.reshape(-1,8100)[np.where(swb[99] == 0)].mean(0)
# tmpstate[1] = tmp.reshape(-1,8100)[np.where(swb[99] == 1)].mean(0)
# tmpstate[2] = tmp.reshape(-1,8100)[np.where(swb[99] == 2)].mean(0)
# tmpstate[3] = tmp.reshape(-1,8100)[np.where(swb[99] == 3)].mean(0)
# states_show(tmpstate)
