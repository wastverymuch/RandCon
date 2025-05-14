import numpy as np


def mtd(x, window=None):

    if x.ndim < 3:
        x = x[None, :]

    n_sub, n_roi, n_tr = x.shape
    y = np.zeros((n_sub, n_tr - 1, n_roi, n_roi))
    z = np.zeros((n_sub,n_tr - window,n_roi,n_roi))
    for sub in range(n_sub):
        data = x[sub].T
        td = data[1:, :] - data[:-1, :]
        data_std = np.std(td, axis=0)
        td_std = td / data_std
        mtd = np.einsum('ti,tj->tij', td_std, td_std)
        y[sub] = mtd
        z[sub] = np.array([np.mean(mtd[i:i+window], axis=0) for i in range(n_tr - window)])
    return z


if __name__ == '__main__':
    pass