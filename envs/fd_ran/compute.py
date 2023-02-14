import mat73 as mat
import numpy as np
import torch


def compute_se_lsfd_mmse(K, D, gki_stat, gki2_stat, F_stat, p_max, all_ues_inf):
    se = np.zeros(K)
    P_U = p_max * torch.ones(K, dtype=torch.float64)
    for k in range(K):
        se[k] = compute_se_lsfd_mmse_k(k, D, gki_stat, gki2_stat, F_stat, P_U, all_ues_inf)
    return se


def compute_se_lsfd_mmse_k(k, D, gki_stat, gki2_stat, F_stat, P_U, all_ues_inf=True, serve_ubs=None):

    if serve_ubs is None:
        serve_ubs = np.where(D[:, k] == 1)[0]

    [M, K] = D.shape
    P_K = P_U[k]

    # Determine which UEs that are served by partially the same set
    # of UBSs as UE k, i.e., the set in (5.15)
    if all_ues_inf:
        ues_inf = torch.tensor(range(K))
    else:
        ues_inf = torch.where(torch.sum(D[serve_ubs, :], 1) >= 1)[0]

    P_S_D = torch.diag(P_U[ues_inf]).type(torch.complex128)
    F_K = torch.diag(F_stat[serve_ubs, k])
    e_g_kk = gki_stat[k, serve_ubs, k].reshape([len(serve_ubs), 1])

    # Compute the first-order entries in the vectors g_{ki}
    e_gki = gki_stat[:, serve_ubs, k][ues_inf].T

    # Compute the second-order entries
    gki2_S = gki2_stat[:, serve_ubs, k][ues_inf]
    e_g_ki2_p = torch.matmul(torch.matmul(e_gki, P_S_D), e_gki.conj().T)
    e_g_ki2_nd = e_g_ki2_p - torch.diag(torch.diag(e_g_ki2_p))
    e_g_kk2_d = torch.diag(torch.sum(torch.matmul(P_S_D, gki2_S), 0))
    e_g_ki2 = e_g_ki2_nd + e_g_kk2_d

    e_g_ki1_k = P_K * torch.matmul(e_g_kk, e_g_kk.conj().T)
    denom_matrix = e_g_ki2 - e_g_ki1_k + F_K

    # Compute the SE achieved with two-layers combine.
    sinr = torch.abs(P_K * torch.matmul(torch.matmul(e_g_kk.conj().T, torch.inverse(denom_matrix)), e_g_kk))
    se = torch.log2(1 + sinr[0, 0])

    return se
