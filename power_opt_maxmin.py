import numpy as np
from absl import logging

def power_opt_max_min(signal, intf, p_max, prelogfactor):
    """

    :param signal: 2-d array shape (K, L)
    :param intf: 4-d array shape (K, L, K, L)
    :param p_max: max transmit power per BS
    :param prelogfactor: prelog factor
    :return: downlink spectral efficiency using the max product power
    allocation. each element [k,j] is the SE of UE k in cell j. Is a 2-d array,
    shape (K, L)
    """

    K = signal.shape[0] # number of UEs
    L = signal.shape[1] # number of cells








def test():
    L = 16 # number of BSs
    K = 10 # UE_per_BS_count
    M = 100 # BS antenna count

    pilot_reuse_factor = 2

    UE_loc_realization_count = 50
    channel_realization_count = 500

    bandwidth = 20e6
    UL_tx_power_UE = 100 # in mW, p
    DL_tx_power_UE = 100 # in mW, rho

    max_DL_tx_power_BS = K * DL_tx_power_UE # maximum dl tx power in mW

    equal_power_alloc = max_DL_tx_power_BS / K * np.ones([K, L]) #rhoEqual
    noise_figure_BS = 7  # in dB
    noise_variance = -174 + 10 * np.log10(bandwidth) + noise_figure_BS

    length_coherence_block = 200 # tau_c

    accuracy_local_scattering = 2

    # angular standard deviation in local scattering model in degrees
    ASD_deg = 10

    for n in range(UE_loc_realization_count):
        logging.info('UE location setup {} of {}'.format(n, UE_loc_realization_count))

        # generate channel statistics
        # channel_gain in dB
        R, channel_gain = channel_compute(L, K, M,
                                           accuracy_local_scattering, ASD_deg)

        # the normalized average channel gain, normalization based on noise
        # power
        channel_gain_over_noise = channel_gain - noise_variance

        for m in range(M):
            # generate channel realizations with estimates and
            # estimation error correlation
            H_hat, C, tau_p, Rscaled, H = channel_estimate(
                R,
                channel_gain_over_noise,
                channel_realization_count,
                M,
                K,
                L,
                UL_tx_power_UE,
                pilot_reuse_factor
            )

            # signal, interference terms of DL SE with the hardening bound in
            # Bjornson's book Theorem 4.6

            signal_MR, interf_MR, \
            signal_RZF, interf_RZF,\
            signal_MMMSE, interf_MMMSE, prelogFactor = computer_SINR_DL(
                H,
                H_hat,
                C,
                length_coherence_block,
                tau_p,
                channel_realization_count,
                M,
                K,
                L,
                UL_tx_power_UE
            )

            del H_hat, C, Rscaled

            # compute the SE with various power allocation
            SE = {}
            power_alloc_methods = ['product_sinr', 'max_min', 'equal']
            precoders = ['MR', 'RZF', 'MMMSE']
            for method in power_alloc_methods:
                SE[method] = {}
                for precode in precoders:
                    logging.info('Power Allocation: {}, Precode: {}'.format(
                        method, precode
                    ))
                    SE[method][precode] = compute_SE_DL(
                        signal=signal_MR,
                        interf=interf_MR,
                        prelogFactor=prelogFactor,
                        power_alloc=method,
                        max_power=max_DL_tx_power_BS,
                        precode_method=precode
                    )



# figure;
# hold on; box on;
#
# plot(sort(SE_MR_maxmin(:)),linspace(0,1,K*L*nbrOfSetups),'k--','LineWidth',1);
# plot(sort(SE_MR_equal(:)),linspace(0,1,K*L*nbrOfSetups),'k-','LineWidth',1);
# plot(sort(SE_MR_maxprod(:)),linspace(0,1,K*L*nbrOfSetups),'k-.','LineWidth',1);
#
# plot(sort(SE_RZF_equal(:)),linspace(0,1,K*L*nbrOfSetups),'b-','LineWidth',1);
# plot(sort(SE_MMMSE_equal(:)),linspace(0,1,K*L*nbrOfSetups),'r','LineWidth',1);
#
# plot(sort(SE_RZF_maxprod(:)),linspace(0,1,K*L*nbrOfSetups),'b-.','LineWidth',1);
# plot(sort(SE_MMMSE_maxprod(:)),linspace(0,1,K*L*nbrOfSetups),'r-.','LineWidth',1);
#
# plot(sort(SE_RZF_maxmin(:)),linspace(0,1,K*L*nbrOfSetups),'b--','LineWidth',1);
# plot(sort(SE_MMMSE_maxmin(:)),linspace(0,1,K*L*nbrOfSetups),'r--','LineWidth',1);
#
# legend('Max-min fairness','Equal power','Max product SINR','Location','SouthEast');
#
# xlabel('SE per UE [bit/s/Hz]');
# ylabel('CDF');


