import numpy as np
from local_scatter import R_local_scattering


def channel_stat_setup(
        L, K, M, accuracy, asd_deg
):
    """
    channel statistics between UE's at random locations and the BS.
    See Emil Bjornson book Sec 4.1.3

    Distances are in meters.

    :param L: # of BS's and cells 
    :param K: # of UE's per cell
    :param M: # of antennas per BS
    :param accuracy: 
    1 - exact correlation matrices from local scattering
    2 - small-angle approximation
    :param asd_deg: angular standard deviation around the nominal angle, 
    in degrees
    :return: R: 5-d array of shape (M, M, K, L, L) with special correlation 
    matrices for all UEs.
    R[:, :, k, j, l] is the correlation matrix for channel b/w UE k in cell j 
    and the BS in cell l. Normalized such that trace(R) = M
    channelGaindB (K, L, L) average channel gain in dBs of all channels.
    R(:,:,k,j,l)*10^(channelGaindB(k,j,l)/10) is the full
    spatial channel correlation matrix.
    """
    side_length = 1000  # square side, in meters
    alpha = 3.76  # pathloss exp

    constant_term = -35.3  # avg. channel gain in dB at the ref. distance 1
    # meter. At exponent set to 3.76, at 1km it's -148.1 dB

    sigma_sf = 10  # standard deviation of shadow fading

    # min. dist b/w BS, UE
    min_UE_BS_dist = 35

    # antenna spacing # of wavelengths
    antenna_spacing = 0.5

    inter_bs_distance = side_length / np.sqrt(L)

    # scatter the BSs

    UEpositions = np.zeros([K, L, 2])
    perBS = np.zeros([L,])

    # normalized spatial correlation matrices
    R = np.zeros([M, M, K, L, L, len(asd_deg)])

    channel_gain = np.zeros([K, L, L])

    for i in range(L):
        # put K UEs in the cell, uniformly. UE's not satisfying the min
        # distance are replaced
        res = []
        while perBS[i] < K:
            UEremaining = K - perBS[i]
            pos = np.random.uniform(-inter_bs_distance/2, inter_bs_distance/2,
                              size=[UEremaining, 2])
            cond = np.linalg.norm(pos, ord=2, axis=1) >= min_UE_BS_dist
            pos = pos[cond] # satisfying min distance w.r.t BS shape (?, 2)
            res.append(pos)
            perBS[i] += pos.shape[0]

        # loop through all BS for cross-channels
        for j in range(L):
            # distance from UE in cell i to BS j, with wrap-around. The
            # shortest distance is considered
            """
            [distancesBSj,whichpos] = min(abs( repmat(UEpositions(:,l),[1 size(BSpositionsWrapped,2)]) - repmat(BSpositionsWrapped(j,:),[K 1]) ),[],2);
            """

            # avg. channel gain w/ large-scale fading model in (2.3),
            # neglecting shadow fading

            channel_gain[:, i, j] = constant_term - alpha * 10 * np.log10(
                distances_BS_j)

            # nominal angle b/w UE k in cell l and BS j
            # generate spatial correlation matrices for channels with local
            # scattering model

            for k in range(K):
                angle_BS_j = np.arctan(UEpositions - BSpositions_wrap[j,
                                                                      whichpos(k)])
                for spr in range(len(asd_deg)):
                    R[:, :, k, i, j, spr] = R_local_scattering(
                        M,
                        angle_BS_j,
                        asd_deg(spr),
                        antenna_spacing,
                        accuracy
                    )

        # all UEs in cell i to generate shadow fading realizations
        for k in range(K):


            # see if another BS has a larger avg. channel gain to the UE than
            # BS i
            while True:
                # generate new shadow fading realizations until all UE's in
                # cell i has its largest avg. channel gain from BS i
                shadowing = sigma_sf * np.random.randn(1, 1, L)
                channel_gain_shadowing = channel_gain[k, i, :] + shadowing
                if channel_gain_shadowing[i] >= max(channel_gain_shadowing):
                    break
            channel_gain[k,i,:] = channel_gain_shadowing

    return R, channel_gain

