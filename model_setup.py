import numpy as np
from local_scatter import R_local_scattering
from utils import randn2

def channel_stat_setup(
        L, K, M, asd_degs, no_BS_per_dim=None, accuracy=2,
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
    :param no_BS_per_dim: array of number of BS' per dimension. If None, use
    (sqrt(L), sqrt(L))
    :return: R: 5-d array of shape (M, M, K, L, L) with special correlation 
    matrices for all UEs.
    R[:, :, k, j, l] is the correlation matrix for channel b/w UE k in cell j 
    and the BS in cell l. Normalized such that trace(R) = M
    channelGaindB (K, L, L) average channel gain in dBs of all channels.
    R(:,:,k,j,l)*10**(channelGaindB(k,j,l)/10) is the full
    spatial channel correlation matrix.
    """
    side_length = 250  # square side, in meters
    alpha = 3.76  # pathloss exp

    constant_term = -35.3  # avg. channel gain in dB at the ref. distance 1
    # meter. At exponent set to 3.76, at 1km it's -148.1 dB

    sigma_sf = 10  # standard deviation of shadow fading

    # min. dist b/w BS, UE
    min_UE_BS_dist = 35

    # antenna spacing # of wavelengths
    antenna_spacing = 0.5
    if no_BS_per_dim is None:
        no_BS_per_dim = np.array([np.sqrt(L), np.sqrt(L)])
    inter_bs_distance = side_length / no_BS_per_dim

    # scatter the BSs
    BS_positions = np.stack(
        np.meshgrid(
            np.arange(inter_bs_distance[0]/2, side_length, inter_bs_distance[
                0]),
            np.arange(inter_bs_distance[1]/2, side_length, inter_bs_distance[
                1]),
            indexing='ij'
        ),
        axis=2).reshape([-1,2])
    # now all the other nine alternatives of the BS locations
    wrap_locations = np.stack(
        np.meshgrid(
            np.array([-side_length, 0, side_length]),
            np.array([-side_length, 0, side_length]),
            indexing='ij'
        ),
        axis=2).reshape([-1,2])
    # for each BS locations, there are 9 possible alternative locations including
    # the original one. Here uses broadcasting to add (9,2) to a (num_BS, 1, 2) to
    # get a (num_BS, 9, 2)
    BS_positions_wrapped = np.expand_dims(BS_positions, axis=1) + wrap_locations

    UEpositions = np.zeros([K, L, 2])
    perBS = np.zeros([L,], dtype=np.int)

    # normalized spatial correlation matrices
    R = np.zeros([M, M, K, L, L, len(asd_degs)], dtype=np.complex128)

    channel_gain = np.zeros([K, L, L])

    for i in range(L):
        # put K UEs in the cell, uniformly. UE's not satisfying the min
        # distance are replaced
        res = []
        while perBS[i] < K:
            UEremaining = K - perBS[i]
            pos = np.random.uniform(-inter_bs_distance/2, inter_bs_distance/2,
                              size=(UEremaining, 2))
            cond = np.linalg.norm(pos, ord=2, axis=1) >= min_UE_BS_dist
            pos = pos[cond, :] # satisfying min distance w.r.t BS shape (?, 2)
            for x in pos:
                res.append(x + BS_positions[i])
            perBS[i] += pos.shape[0]
        UEpositions[:, i, :] = np.array(res)

        # loop through all BS for cross-channels
        for j in range(L):
            # distance between all UEs in cell i to BS j, considering wrap-around.
            # The shortest of the 9 position is returned
            dist_ue_i_j = np.linalg.norm(np.expand_dims(UEpositions[:, i], axis=1) - BS_positions_wrapped[j, :, :], axis=2)
            dist_bs_j = np.min(dist_ue_i_j, axis=1)
            which_pos = np.argmin(dist_ue_i_j, axis=1)

            # avg. channel gain w/ large-scale fading model in (2.3),
            # neglecting shadow fading

            channel_gain[:, i, j] = constant_term - alpha * 10 * np.log10(
                dist_bs_j)

            # nominal angle b/w UE k in cell l and BS j
            # generate spatial correlation matrices for channels with local
            # scattering model

            for k in range(K):
                vec_ue_bs = UEpositions[k, i] - BS_positions_wrapped[j, which_pos[k]]
                angle_BS_j = np.arctan2(vec_ue_bs[1], vec_ue_bs[0])
                for spr, asd_deg in enumerate(asd_degs):
                    R[:, :, k, i, j, spr] = R_local_scattering(
                        M,
                        angle_BS_j,
                        asd_deg,
                        antenna_spacing,
                        accuracy=accuracy
                    )

        # all UEs in cell i to generate shadow fading realizations
        for k in range(K):
            # see if another BS has a larger avg. channel gain to the UE than
            # BS i
            while True:
                # generate new shadow fading realizations until all UE's in
                # cell i has its largest avg. channel gain from BS i
                shadowing = sigma_sf * randn2(L)
                channel_gain_shadowing = channel_gain[k, i] + shadowing
                if channel_gain_shadowing[i] >= np.max(channel_gain_shadowing):
                    break
            channel_gain[k,i,:] = channel_gain_shadowing

    return R, channel_gain

