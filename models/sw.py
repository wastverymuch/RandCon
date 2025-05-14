import numpy as np

def sliding_windows(x,window_length = 20,step = 1):
    if x.shape.__len__() < 3:
        x=x[None,:]
    n_sub,n_roi,n_tr = x.shape
    n_window = int((n_tr - window_length) / step) + 1
    y=np.zeros((n_sub,n_window,n_roi,n_roi))
    for i in range(n_sub):
        for j in range(n_window):
            y[i,j] = np.corrcoef(x[i,:,j * step:j * step + window_length])
    return y

if __name__ == '__main__':
    pass