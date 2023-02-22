
import numpy as np
import torch

from .compute import compute_se_lsfd_mmse_k
from .eutils import db2pow


def access_pilot(M, K, Gain, tau_p, gain_thr=40):
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
    pilots = 1000*torch.ones(K, dtype=torch.int)
    D_C = torch.zeros([M, K], dtype=torch.int)
    masterAPs = torch.zeros(K, dtype=torch.int)
    master_val = torch.zeros(K, dtype=torch.float64)
    master_count = torch.zeros(M, dtype=torch.int)  # Store the count that the bs served as.

    # Set threshold for when a non-master AP decides to serve a UE
    threshold = -40  # dB

    for k in range(K):

        while True:
            # channel condition
            master = torch.argmax(Gain[:, k])

            # If the count is larger than the number of pilots, than it will not serve as master APs for other users.
            if master_count[master] < tau_p:

                master_count[master] += 1

                D_C[master, k] = 1
                masterAPs[k] = master
                master_val[k] = Gain[master, k]

                # Assign orthogonal pilots to the first tau_p UEs
                if k < tau_p:

                    pilots[k] = k

                else:  # Assign pilot for remaining UEs

                    # Compute received power from ue to the master AP from each pilot
                    pilot_inf = torch.zeros(tau_p, 1)
                    for t in range(tau_p):
                        pilot_inf[t] = torch.sum(db2pow(Gain[master, (pilots == t)]))

                    # np.where the pilot with the least receiver power
                    pilots[k] = torch.argmin(pilot_inf)

                break

            else:

                Gain[master, k] = -1e10
                continue

    # TODO: =================== BS will serve user with the best signal.========================
    # Each AP serves the UE with the strongest channel condition on each of
    # the pilots where the AP isn't the master AP, but only if its channel
    # is not too weak compared to the master AP
    for m in range(M):

        for t in range(tau_p):

            pilot_ues = torch.where(t == pilots)[0]
            # Users with pilot t.

            # If the AP is not serve any user with pilot t.
            if (sum(D_C[m, pilot_ues]) == 0) & (len(pilot_ues) > 0) & (D_C[m].sum() < tau_p):

                # np.where the UE with pilot t with the best channel
                [gainValue, UEindex] = torch.max(Gain[m, pilot_ues], dim=0)

                # Serve this UE if the channel is at most "threshold" weaker
                # than the master AP's channel
                # [gainValue Gain(masterAPs(pilotUEs(UEindex)), pilotUEs(UEindex),n)]
                gain_b = Gain[masterAPs[pilot_ues[UEindex]], pilot_ues[UEindex]]

                if gainValue - gain_b >= threshold:
                    D_C[m, pilot_ues[UEindex]] = 1

    # TODO: ======================== User centric access candidate.============================
    # ues_idx_sig = torch.argsort(master_val)
    # for i in range(K):
    #     k = ues_idx_sig[i]
    #     pass

    return D_C, pilots


def semvs_associate(se_inc_thr, M, K, K_T, D_C, g_stat, g2_stat, F_stat, p_max):
    """
    > The function `semvs_associate` is used to associate the users with the UBSs.

    :param se_inc_thr: The minimum improvement in SE that is required to add a new UBS to the set of
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
    D_S = torch.zeros([M, K], dtype=torch.int)
    power = p_max * torch.ones(K)

    # Determine the set of candidate serving ubs for all users.
    M_k_set = [[] for _ in range(K)]    # Service UBS set for user K.
    K_m_set = [[] for _ in range(M)]    # Service User set for UBS m.

    # Setup a small se, so if adding a new UBS, then 'phi' is large.
    se_opt = 1e-20 * torch.ones(K)

    M_C_set = [[] for _ in range(K)]

    # Candidate service UBS set for any user k.
    n_can_ubs = torch.zeros(K, dtype=torch.int)

    # TAG: Statistics of numbers for candidate service ubs for user k.
    for k in range(K):
        can_ubs_k = torch.where(D_C[:, k] == 1)[0].flatten()  # ubs index.
        M_C_set[k] = can_ubs_k
        n_can_ubs[k] = len(can_ubs_k)

    # TAG: Sort the users according to number of acceptable service ubs.
    ue_sort_idx = torch.argsort(n_can_ubs)
    se_inc = torch.zeros([M, K])

    while True:

        # INFO: Iterate every user .
        for i in range(K):
        
            k = ue_sort_idx[i]
            can_ubs_k = M_C_set[k]
            n_can_ubs_k = len(can_ubs_k)

            if n_can_ubs_k <= 0:
                continue

            opt_se_k, opt_ubs_k, del_ubs_k = choose_ubs(se_inc_thr, k, can_ubs_k, M_k_set[k], se_opt[k],
                                                        power, D_C, g_stat, g2_stat, F_stat)

            # INFO: Update records.
            if (opt_ubs_k != -1) & (len(K_m_set[opt_ubs_k]) < K_T):

                se_inc[opt_ubs_k, k] = opt_se_k - se_opt[k]
                se_opt[k] = opt_se_k

                M_k_set[k].append(opt_ubs_k)
                K_m_set[opt_ubs_k].append(k)

            # INFO: Delete the ubs.
            if len(del_ubs_k) > 0:
                for j in range(len(del_ubs_k)):
                    M_C_set[k] = M_C_set[k][M_C_set[k] != del_ubs_k[j]]
                    if n_can_ubs[k] - 1 < 0:
                        opt_se_k, opt_ubs_k, del_ubs_k = choose_ubs(se_inc_thr, k, can_ubs_k, M_k_set[k], se_opt[k],
                                                        power, D_C, g_stat, g2_stat, F_stat)
                    n_can_ubs[k] = n_can_ubs[k] - 1

        # INFO: If there is no candidate ubs.
        if sum(n_can_ubs) == 0:
            break

    for k in range(K):
        D_S[M_k_set[k], k] = 1

    return D_S, se_inc


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

        serve_ubs_ext = serve_ubs.copy()
        m_idx = can_ubs_set[j]  # observe ubs index.

        # Expand set to evaluate.
        serve_ubs_ext.append(m_idx)    # UBS indexes

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
    elif m_opt != -1:
        del_ubs.append(m_opt)

    return opt_se, m_opt, del_ubs
