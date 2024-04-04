import matplotlib.pyplot as plt
import os


def plot_competition_figures(intrf_sig_names, all_sinr_db, mse_loss_model, mse_loss, ber_model, ber,
                             soi_type, prefix=''):
    os.makedirs('figures', exist_ok=True)
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]

    fig_mse, axs_mse = plt.subplots(2, 2, figsize=(10, 8))
    fig_ber, axs_ber = plt.subplots(2, 2, figsize=(10, 8))

    for i, sig_name in enumerate(intrf_sig_names):
        # Plot MSE
        axs_mse[rows[i], cols[i]].semilogy(
            all_sinr_db, mse_loss[i], label='No mitigation', linewidth=3, linestyle='--', c='b', marker='o')
        axs_mse[rows[i], cols[i]].semilogy(
            all_sinr_db, mse_loss_model[i], label='Model', linewidth=3, linestyle='--', c='r', marker='o')
        axs_mse[rows[i], cols[i]].set_title(f'{soi_type}_{sig_name}', fontsize=14)
        axs_mse[rows[i], cols[i]].grid()
        axs_mse[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
        if cols[i] == 0:
            axs_mse[rows[i], cols[i]].set_ylabel('Mean Squared Error', fontsize=12)
        axs_mse[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
        axs_mse[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

        # Plot BER
        axs_ber[rows[i], cols[i]].semilogy(
            all_sinr_db, ber[i], label='No mitigation', linewidth=3, linestyle='--', c='b', marker='o')
        axs_ber[rows[i], cols[i]].semilogy(
            all_sinr_db, ber_model[i], label='Model', linewidth=3, linestyle='--', c='r', marker='o')
        axs_ber[rows[i], cols[i]].set_title(f'{soi_type}_{sig_name}', fontsize=14)
        axs_ber[rows[i], cols[i]].grid()
        axs_ber[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
        if cols[i] == 0:
            axs_ber[rows[i], cols[i]].set_ylabel('Bit error rate', fontsize=12)
        axs_ber[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
        axs_ber[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

    fig_mse.subplots_adjust(hspace=0.75)
    fig_mse.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2,
                   labels=["No mitigation", "Model"], fontsize=14)
    fig_ber.subplots_adjust(hspace=0.75)
    fig_ber.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2,
                   labels=["No mitigation", "Model"], fontsize=14)
    # Save the figures
    fig_mse.savefig(f'figures/{prefix}_competition_mse.png', dpi=800, bbox_inches='tight')
    fig_ber.savefig(f'figures/{prefix}_competition_ber.png', dpi=800, bbox_inches='tight')
    plt.show()
