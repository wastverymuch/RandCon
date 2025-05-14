import numpy as np
from scipy.signal import hilbert


def phase(all_data):
    if all_data.ndim == 2:
        all_data = all_data[None, :]

    n_sub, n_roi, n_tr = all_data.shape
    need = np.zeros((n_sub, n_tr, n_roi, n_roi))

    for sub in range(n_sub):
        analytic_signal = hilbert(all_data[sub], axis=1)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal), axis=1)
        for t in range(n_tr):
            phase_diff_matrix = instantaneous_phase[:, t][:, None] - instantaneous_phase[:, t]
            V_t = np.cos(phase_diff_matrix)
            need[sub, t] = V_t

    return need
if __name__ == '__main__':

    pass