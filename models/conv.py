import torch
from utils import *
# from utils import corshow
from sklearn import metrics
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 此代码用于大规模测试通道数，正确答案密度，噪声和卷积核形状的影响

if __name__ == '__main__':
    conv_data_path = '/longlab/duan/pycharm_programs/formal_project/osl/conv/'
    out_channels = [5,10,20,40,80,160]
    sizes = [1,5,10,15]
    noises = [0,0.2,0.4,0.6,0.8,1.0]
    aris31 = np.zeros((len(out_channels), len(sizes),len(noises)))
    aris33 = np.zeros((len(out_channels), len(sizes),len(noises)))
    for channel in out_channels:
        conv33 = nn.Conv2d(1,channel,3,1,1,1,1,False,'replicate')
        conv31 = nn.Conv2d(1,channel,(3,1),1,(1,0),1,1,False,'replicate')
        for size in sizes:
            data = np.load((conv_data_path + f'{size}size_05_200sub_90_1200_321seed.npz'))
            tc = data['tc'][:,None]
            tc = torch.from_numpy(tc).float()
            index = data['index']
            ground_truth = data['ground_truth']
            for noise in noises:
                timea = time.time()
                tc += noise * torch.randn(200,1,90,1200)
                need31 = corshow(conv31(tc[:10]).transpose(-1,-3).reshape(-1,90,channel))
                need33 = corshow(conv33(tc[:10]).transpose(-1,-3).reshape(-1,90,channel))
                timeb = time.time()
                cen31, ind31 = faisscluster(need31, 100)
                cen33, ind33 = faisscluster(need33, 100)
                timec = time.time()
                ari = []
                for t in range(100):
                    ari.append(metrics.adjusted_rand_score(ind31[t],index[:10].reshape(-1)))
                aris31[out_channels.index(channel),sizes.index(size),noises.index(noise)] = np.max(ari)
                ari2 = []
                for t in range(100):
                    ari2.append(metrics.adjusted_rand_score(ind33[t],index[:10].reshape(-1)))
                aris33[out_channels.index(channel),sizes.index(size),noises.index(noise)] = np.max(ari2)
                timed = time.time()
    np.savez('./exp_ari.npz',aris31=aris31,aris33=aris33)
    pass



def conv(x,out_channels = 10,kernal_shape = '33',windows_length = 3,padding = True):
    from utils import corshow
    if kernal_shape == '33':
        con = nn.Conv2d(1,out_channels,3,1,1,1,1,False)
    elif kernal_shape == '31':
        con = nn.Conv2d(1, out_channels, (3,1), 1, (1,0), 1, 1, False)
    elif kernal_shape == '13':
        con = nn.Conv2d(1, out_channels, (1, 3), 1, (0, 1), 1, 1, False)
    if windows_length != 3 or padding == False:
        con = nn.Conv2d(1, out_channels, (1,windows_length), 1, 0, 1, 1, False)
    if x.ndim < 3:
        x = x[None, :]
    n_sub,n_roi,n_tr = x.shape
    x = torch.from_numpy(x[:,None]).float()
    y = con(x)
    y = corshow(y.transpose(-1,-3).reshape(-1,n_roi,out_channels))
    y = y.reshape(n_sub,-1,n_roi,n_roi)
    return y
