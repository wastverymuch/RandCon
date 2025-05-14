from utils import *
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conv(x,out_channels = 10,windows_length = 3,padding = True):

    con = nn.Conv2d(1, out_channels, (1, 3), 1, (0, 1), 1, 1, False,'replicate')
    if windows_length != 3 or padding == False:
        con = nn.Conv2d(1, out_channels, (1,windows_length), 1, 0, 1, 1, False,'replicate')
    if x.ndim < 3:
        x = x[None, :]
    n_sub,n_roi,n_tr = x.shape
    x = torch.from_numpy(x[:,None]).float()
    y = con(x)
    y = corshow(y.transpose(-1,-3).reshape(-1,n_roi,out_channels))
    y = y.reshape(n_sub,-1,n_roi,n_roi)
    return y


if __name__ == '__main__':
    pass
