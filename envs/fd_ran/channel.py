import mat73 as mat
import numpy as np
import torch
from scipy import linalg

from .eutils import db2pow, pow2db


def environment(network, ues_pos, ubs_pos, N, use_scatter):

    # If square is true, then width means the width of square.
    # else, it means the ISD for inter-BSs.

    M = len(ubs_pos)
    K = len(ues_pos)

    # Communication bandwidth
    B = 20e6

    if network == 'cell':
        heigh_2 = torch.tensor(35 ^ 2)
    else:
        heigh_2 = torch.tensor(100)

    # Noise figure ( in dB)
    noiseFigure = 7

    # Compute noise power
    noise_variance = -174 + 10 * np.log10(B) + noiseFigure

    # Pathloss exponent
    alpha = 3.76

    # Standard deviation of shadow fading
    sigma_sf = 4

    # Average channel gain in dB at a reference distance of 1 meter. Note that
    # -35.3 dB corresponds to - 148.1 dB at 1 km, using pathloss exponent 3.76

    constantTerm = -35.3

    # Define the antenna spacing ( in number of wavelens)
    antennaSpacing = 1/2
    # Half wavelen distance

    # Angular standard deviation around the nominal angle(measured in degrees)
    ASDdeg = 20

    # Prepare to save results
    gainOverNoisedB = torch.zeros([M, K], dtype=torch.double)
    R = torch.zeros([N, N, M, K], dtype=torch.complex128)
    # # Go through all setups
    dis_vec = (ues_pos.expand([K, M]) - ubs_pos.expand([K, M])).T

    # Compute nominal angle between UE k and AP l
    angle_bs2ue = torch.angle(dis_vec)
    distances = torch.sqrt(torch.square(torch.abs(dis_vec) + heigh_2))
    radom_shadow = torch.randn([M, K], dtype=torch.double)
    # Compute the channel gain divided by noise power
    gainOverNoisedB = constantTerm - alpha * 10 * \
        torch.log10(distances) + sigma_sf * radom_shadow - noise_variance

    for k in range(K):

        # Go through all APs
        if use_scatter:
            for m in range(M):
                # Generate normalized spatial correlation matrix using the local
                # scattering model
                scatter_ = Rlocalscattering(N, angle_bs2ue[m, k], ASDdeg, antennaSpacing).conj()
                R[:, :, m, k] = db2pow(gainOverNoisedB[m, k]) * scatter_

        else:
           # Go through all APs
            for m in range(M):
                R[:, :, m, k] = db2pow(gainOverNoisedB[m, k])

    return [R, gainOverNoisedB]


def Rlocalscattering(M, theta, ASDdeg, antennaSpacing=0.5, distribution='Gaussian', n_interger=100):
    # Generate the spatial correlation matrix for the local scattering model,
    # defined in (2.23) for different angular distributions.
    #
    # INPUT:
    # M = Number of antennas
    # theta = Nominal angle
    # ASDdeg = Angular standard deviation around the nominal angle
    #                 (measured in degrees)
    # antennaSpacing = (Optional) Spacing between antennas ( in wavelengths)
    # distribution = (Optional) Choose between 'Gaussian', 'Uniform', and
    #                'Laplace' angular distribution. Gaussian is default
    #
    # OUTPUT:
    # R = M x M spatial correlation matrix

    # Compute the ASD in radians based on input
    ASD = torch.tensor(ASDdeg) * torch.pi / 180

    # The correlation matrix has a Toeplitz structure, so we only need to
    # compute the first row of the matrix
    firstRow = torch.zeros(M, dtype=torch.complex128)

    # Go through all the columns of the first row
    for column in range(M):
       # Distance from the first antenna
        distance = antennaSpacing * column

        # For Gaussian angular distribution
        if distribution == 'Gaussian':

            # Define integrand of (2.23)
            # Delta = 1
            # a = exp(1j*2*pi*distance*sin(theta+Delta))
            # b = exp(-Delta.^2/(2*ASD^2));
            # c = sqrt(2*pi)*ASD
            # y = a*b/c

            x = torch.linspace(-20 * ASD, 20 * ASD, n_interger)
            y = torch.exp(1j * 2 * torch.pi * distance * torch.sin(theta + x)) * \
                torch.exp(-torch.square(x) / (2 * torch.square(ASD))) / (np.sqrt(2 * torch.pi) * ASD)

        # For uniform angular distribution
        elif distribution == 'Uniform':

            # Set the upper and lower limit of the uniform distribution
            limits = torch.sqrt(3) * ASD
            x = torch.linspace(-limits, limits, n_interger).cuda()

            # Define integrand of(2.23)
            y = torch.exp(1j * 2 * torch.pi * distance * torch.sin(theta + x)) / (2 * limits)

        # For Laplace angular distribution
        elif distribution == 'Laplace':

            x = torch.linspace(-20 * ASD, 20 * ASD, n_interger).cuda()

            # Set the scale parameter of the Laplace distribution
            LaplaceScale = ASD / torch.sqrt(2)

            # Define integrand of(2.23)
            y = torch.exp(1j * 2 * torch.pi * distance * torch.sin(theta + x)) * \
                torch.exp(-abs(x) / LaplaceScale) / (2 * LaplaceScale)

        # Compute the integral in (2.23) by including 20 standard deviations
        firstRow[column] = torch.trapezoid(y, x)

    # Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R = linalg.toeplitz(firstRow)

    return R


def toeplitz(c, r=None):

    r = c.conjugate()

    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    vals = torch.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return torch.as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


def chl_estimate(R, R_sqrtm, N_chs, M, N, K, tau_p, pilotIndex, power):

    H = torch.randn([M * N, N_chs, K], dtype=torch.complex128) + 1j * \
        torch.randn([M * N, N_chs, K], dtype=torch.complex128)

    # Go through all channels and apply the spatial correlation matrices
    for l in range(M):
        for k in range(K):

           # Apply correlation to the uncorrelated channel realizations
            Rsqrt = R_sqrtm[:, :, l, k]
            H[l * N: (l+1) * N, :, k] = np.sqrt(0.5) * torch.matmul(Rsqrt, H[l * N: (l+1) * N, :, k])

    # Perform channel estimation
    # Store identity matrix of size N x N
    eyeN = torch.eye(N)

    # Generate realizations of normalized noise
    Np = np.sqrt(0.5) * (torch.randn([N, N_chs, M, tau_p])
                         + 1j * torch.randn([N, N_chs, M, tau_p]))

    # Prepare to store results
    Hhat = torch.zeros([M * N, N_chs, K], dtype=torch.complex128)
    B = torch.zeros_like(R, dtype=torch.complex128)
    C = torch.zeros_like(R, dtype=torch.complex128)

    # Go through all APs
    for l in range(M):
       # Go through all pilots
        for t in range(tau_p):

            # Compute processed pilot signal for all UEs that use pilot t
            # according to(4.4) with an additional scaling factor \sqrt{tau_p}
            yp = np.sqrt(power) * tau_p * torch.sum(H[l * N: (l+1) * N, :,
                                                      t == pilotIndex], 2) + np.sqrt(tau_p) * Np[:, :, l, t]

            # Compute the matrix in (4.6) that is inverted in the MMSE estimator
            # in (4.5)
            PsiInv = (power * tau_p * torch.sum(R[:, :, l, t == pilotIndex], dim=2) + eyeN)

            # Go through all UEs that use pilot t
            ue_idx = torch.where(t == pilotIndex)[0]

            for k in ue_idx:

                # Compute the MMSE estimate
                RPsi = torch.matmul(R[:, :, l, k], torch.inverse(PsiInv))
                Hhat[l * N: (l+1) * N, :, k] = np.sqrt(power) * torch.matmul(RPsi, yp)
                # Compute the spatial correlation matrix of the estimate
                # according to(4.7)
                B[:, :, l, k] = power * tau_p * torch.matmul(RPsi, R[:, :, l, k])
                # Compute the spatial correlation matrix of the estimation
                # error according to(4.9)
                C[:, :, l, k] = R[:, :, l, k] - B[:, :, l, k]

    return Hhat, H, C


def channel_statistics(Hhat, H, D, C, num_chs, M, N, K, p_max, all_ues_inf=True, sigma2=1):

    P_U = p_max * torch.ones(K)
    # Store the N x N identity matrix
    eyeN = sigma2 * torch.eye(N)
    # Scale C by power coefficients
    Cp = torch.zeros_like(C, dtype=torch.complex128)

    for k in range(K):
        Cp[:, :, :, k] = P_U[k] * C[:, :, :, k]

    g_stat = torch.zeros([K, K, M], dtype=torch.complex128)    # Service Users -- Other Users -- UBS
    g2_stat = torch.zeros([K, K, M], dtype=torch.complex128)  # Service Users -- Other Users -- UBS
    F_stat = torch.zeros([M, K], dtype=torch.complex128)

    # Compute scaling factors for combining/precoding

    # Go through all channel realizations
    for n in range(num_chs):

       # Go through all APs
        for m in range(M):
            # Extract channel realizations from all UEs to AP m
            H_m = torch.reshape(H[m * N: (m+1) * N, n, :], [N, K])

            # Extract channel estimate realizations from all UEs to AP m
            H_m_e = torch.reshape(Hhat[m * N: (m+1) * N, n, :], [N, K])

            # Obtain the statistical matrices used for
            # computing partial combining/precoding schemes
            if all_ues_inf:
                ues_inf = torch.tensor(range(K))
            else:
                # Extract which UEs are served by AP m
                ues_inf = torch.where(D[m, :] == 1)[0]

            # The channel that ubs consider for interference.
            H_inf = H_m_e[:, ues_inf]
            P_inf = torch.diag(P_U[ues_inf]).type(torch.complex128)

            # The channel that ubs observations.
            ues_obs = (D[m, :] == 1)  # Extract which UEs are served by AP m
            H_obs = H_m_e[:, ues_obs]
            P_obs = torch.diag(P_U[ues_obs]).type(torch.complex128)

            # Compute C for w.
            Cp_obs = torch.sum(Cp[:, :, m, ues_inf], 2).reshape([N, N])

            # Compute LP-MMSE combining
            w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(torch.matmul(H_inf, P_inf), H_inf.conj().T) +
                                                        Cp_obs + eyeN), H_obs), P_obs)

            # Compute realizations of the terms inside the expectations
            # We should save w'*H_m.
            g_stat[ues_obs, :, m] = g_stat[ues_obs, :, m] + torch.matmul(w.conj().T, H_m) / num_chs
            g2_stat[ues_obs, :, m] = g2_stat[ues_obs, :, m] + \
                torch.square(torch.abs(torch.matmul(w.conj().T, H_m))) / num_chs
            F_stat[m, ues_obs] = F_stat[m, ues_obs] + sigma2 * torch.square(torch.norm(w, dim=0)) / num_chs

    # Permute the arrays that consist of the expectations that appear in Theorem
    g_stat = torch.permute(g_stat, [1, 2, 0])  # Other Users -- UBS -- Service Users
    g2_stat = torch.permute(g2_stat, [1, 2, 0])  # Other Users -- UBS -- Service Users

    return g_stat, g2_stat, F_stat
