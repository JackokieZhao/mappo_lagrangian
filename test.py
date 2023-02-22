import numpy as pkl

from config import get_config
from envs.fd_ran.associate import semvs_associate
from envs.fdran_env import FdranEnv

# if __name__ == '__main__':
#     sce_idx = 1
#     device = "cpu"
#     parser = get_config()
#     cfgs = parser.parse_known_args()[0]

#     fd = FdranEnv(cfgs, sce_idx, device)
if __name__ == '__main__':
    se_inc_thr, M, K, K_T, D_C, g_stat, g2_stat, F_stat, p_max = pkl.load(open('test.pkl', 'rb'), allow_pickle=True)
    D_S, se_inc = semvs_associate(se_inc_thr, M, K, K_T, D_C, g_stat, g2_stat, F_stat, p_max)
    print()
