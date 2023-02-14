import time

import mat73 as mat
import matlab.engine as engine
import numpy as np
import torch

from associate import access_pilot, semvs_associate
from channel import channel_statistics, chl_estimate, environment
from compute import compute_se_lsfd_mmse
from positions import gen_ues_pos, square_deployment
from show import show_bs_ues


def gen_channel():
    # R, Gain, R_sqrtm
    K = 50
    use_scatter = True
    p_max = 200
    N_chs = 50
    width = 1000
    # # Square simulation.
    # Generate ubs and ues positions with cellular networks.
    M = 16
    N = 4
    tau_p = 10
    se_imp_thr = 0.01

    eng = engine.start_matlab()

    for i in range(100):
        eng.main()

        data = mat.loadmat('./res.mat')
        R = torch.tensor(data['R'])
        Gain = torch.tensor(data['gain'])
        R_sqrtm = torch.tensor(data['R_sqrt'])

        # candidate ubs and pilot allocation.
        [D_C, pilot] = access_pilot(M, K, Gain, tau_p)

        Hhat, H, C = chl_estimate(R, R_sqrtm, N_chs, M, N, K, tau_p, pilot, p_max)

        Hhat = torch.tensor(data['Hhat'])
        H = torch.tensor(data['H'])
        C = torch.tensor(data['C'])

        # Statistics for channels.
        [gki_stat, gki2_stat, F_stat] = channel_statistics(Hhat, H, D_C, C, N_chs, M, N, K, p_max)

        # # Determine the access matrix for FD-RAN.
        D_S = semvs_associate(se_imp_thr, M, K, tau_p, D_C, gki_stat, gki2_stat, F_stat, p_max)

        # Compute spectrum efficiency.
        se = compute_se_lsfd_mmse(K, D_S, gki_stat, gki2_stat, F_stat, p_max, True)

        print(np.sum(se - data['se']))

    return D_S, se


if __name__ == '__main__':
    gen_channel()
