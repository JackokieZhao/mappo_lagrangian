
import numpy as np
import torch

from .compute import compute_se_lsfd_mmse_k
from .eutils import db2pow


def access_pilot(M, K, Gain, tau_p):
    """
    The function takes in the channel gain matrix, the number of users, the number of APs, the number of
    orthogonal pilots, and the threshold for when a non-master AP decides to serve a UE. It then assigns
    orthogonal pilots to the first tau_p UEs, and then assigns pilots for the remaining UEs. Each AP
    serves the UE with the strongest channel condition on each of the pilots where the AP isn't the
    master AP, but only if its channel is not too weak compared to the master AP

    :param M: number of APs
    :param K: number of users
    :param Gain: M x K matrix of channel gains
    :param tau_p: number of orthogonal pilots
    :return: D_fd is a matrix of size MxK, where M is the number of APs and K is the number of UEs.
    """
    pilot_fd = 1000*torch.ones(K, dtype=torch.int)
    D_fd = np.zeros([M, K], dtype=np.int)
    masterAPs = np.zeros(K, dtype=np.int)
    # Set threshold for when a non-master AP decides to serve a UE
    threshold = -40
    # dB

    for k in range(K):

        # channel condition
        master = torch.argmax(Gain[:, k])
        D_fd[master, k] = 1
        masterAPs[k] = master

        # Assign orthogonal pilots to the first tau_p UEs
        if k < tau_p:

            pilot_fd[k] = k

        else:  # Assign pilot for remaining UEs

            # Compute received power from ue to the master AP from each pilot
            pilot_inf = torch.zeros(tau_p, 1)
            for t in range(tau_p):
                pilot_inf[t] = torch.sum(db2pow(Gain[master, (pilot_fd == t)]))

            # np.where the pilot with the least receiver power
            bestpilot = torch.argmin(pilot_inf)
            pilot_fd[k] = bestpilot

    # Each AP serves the UE with the strongest channel condition on each of
    # the pilots where the AP isn't the master AP, but only if its channel
    # is not too weak compared to the master AP
    for m in range(M):

        for t in range(tau_p):

            pilotUEs = torch.where(t == pilot_fd)[0]
            # Users with pilot t.

            # If the AP is not a master AP with pilot t.
            if (sum(D_fd[m, pilotUEs]) == 0) & (len(pilotUEs) > 0):

                # np.where the UE with pilot t with the best channel
                [gainValue, UEindex] = torch.max(Gain[m, pilotUEs], dim=0)

                # Serve this UE if the channel is at most "threshold" weaker
                # than the master AP's channel
                # [gainValue Gain(masterAPs(pilotUEs(UEindex)), pilotUEs(UEindex),n)]
                gain_b = Gain[masterAPs[pilotUEs[UEindex]], pilotUEs[UEindex]]

                if gainValue - gain_b >= threshold:
                    D_fd[m, pilotUEs[UEindex]] = 1

    return D_fd, pilot_fd


def semvs_associate(se_imp_thr, M, K, K_T, D_C, g_stat, g2_stat, F_stat, p_max):
    """
    > The function `semvs_associate` is used to associate the users with the UBSs.

    :param se_imp_thr: The minimum improvement in SE that is required to add a new UBS to the set of
    serving UBSs
    :param M: number of UBSs
    :param K: number of users
    :param K_T: The maximum number of users that can be served by a UBS
    :param D_C: The set of candidate UBSs for each user
    :param g_stat: the channel gain matrix
    :param g2_stat: the channel gain matrix
    :param F_stat: the channel gain matrix
    :param p_max: maximum power of each UBS
    :return: the service association matrix.
    """

    # Compute the candidate service service.
    D_S = np.zeros([M, K], dtype=np.int)
    power = p_max * torch.ones(K)

    # Determine the set of candidate serving ubs for all users.
    M_k_set = [[] for _ in range(K)]    # Service UBS set for user K.
    K_m_set = [[] for _ in range(M)]    # Service User set for UBS m.

    # Setup a small se, so if adding a new UBS, then 'phi' is large.
    se_opt = 1e-5 * torch.ones(K, 1)

    M_C_set = [[] for _ in range(K)]

    # Candidate service UBS set for any user k.
    n_can_ubs = np.zeros(K, dtype=np.int)

    # TAG: Statistics of numbers for candidate service ubs for user k.
    for k in range(K):
        can_ubs_k = list(np.argwhere(D_C[:, k] == 1).flatten())  # ubs index.
        M_C_set[k] = can_ubs_k
        n_can_ubs[k] = len(can_ubs_k)

    # TAG: Sort the users according to number of acceptable service ubs.
    ue_sort_idx = np.argsort(n_can_ubs)

    while True:

        # INFO: Iterate every user .
        for i in range(K):

            k = ue_sort_idx[i]
            can_ubs_k = M_C_set[k]
            n_can_ubs_k = len(can_ubs_k)

            if n_can_ubs_k <= 0:
                continue

            opt_se_k, opt_ubs_k, del_ubs_k = choose_ubs(se_imp_thr, k, can_ubs_k, M_k_set[k], se_opt[k],
                                                        power, D_C, g_stat, g2_stat, F_stat)

            # INFO: Update records.
            if (opt_ubs_k != -1) & (len(K_m_set[opt_ubs_k]) < K_T):
                se_opt[k] = opt_se_k
                M_k_set[k].append(opt_ubs_k)
                K_m_set[opt_ubs_k].append(k)

            # INFO: Delete the ubs.
            if len(del_ubs_k) > 0:
                for i in range(len(del_ubs_k)):
                    M_C_set[k].remove(del_ubs_k[i])
                    n_can_ubs[k] = n_can_ubs[k] - 1

        # INFO: If there is no candidate ubs.
        if sum(n_can_ubs) == 0:
            break

    for k in range(K):
        D_S[M_k_set[k], k] = 1

    return D_S


def choose_ubs(inc_thr, k, can_ubs_set, serve_ubs, cu_se_opt, power, D, g_stat, g2_stat, F_stat):
    """
    > The function takes in a set of candidate UBSs, and returns the optimal UBS

    :param inc_thr: the threshold for the increase in SE
    :param k: the user index
    :param can_ubs_set: the set of candidate UBSs
    :param serve_ubs: the UBSs that are currently serving the user
    :param cu_se_opt: the current optimal se
    :param power: the power of the user
    :param D: the distance matrix
    :param g_stat: the channel first-order statistics matrix
    :param g2_stat: the second order statistics of the channel
    :param F_stat: the F matrix
    """
    n_can_ubs = len(can_ubs_set)

    opt_se = cu_se_opt  # The optimal se.
    m_opt = -1   # optimal link ubs.
    del_ubs = []

    for j in range(n_can_ubs):

        m_idx = can_ubs_set[j]  # observe ubs index.

        # Expand set to evaluate.
        serve_ubs_ext = np.append(serve_ubs, m_idx).astype(np.int64)    # UBS indexes

        # serve_ubs_ext = np.where(D(: , k) == 1)
        se_tmp = compute_se_lsfd_mmse_k(k, D, g_stat, g2_stat, F_stat, power, True, serve_ubs_ext)

        # Judge whether m_idx is the best candidate ubs.
        if se_tmp > opt_se:
            opt_se = se_tmp
            m_opt = m_idx

        if (se_tmp - cu_se_opt) / cu_se_opt < inc_thr:
            del_ubs.append(m_idx)

    # If the optimal ubs cannot meet the se incresement threshold.
    if m_opt in del_ubs:
        m_opt = -1
    else:
        del_ubs.append(m_opt)

    return opt_se, m_opt, del_ubs
