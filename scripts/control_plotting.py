import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cont_results_dict_1',
                    type=str,
                    required=True)

parser.add_argument('--cont_results_dict_2',
                    type=str,
                    required=True,)

args = parser.parse_args()
cont_results_1 = json.load(open(args.cont_results_dict_1))
cont_results_2 = json.load(open(args.cont_results_dict_2))

plt.style.use('seaborn-v0_8')


if cont_results_1['sim'] == 'TwoTankControl':
    t_range = np.arange(0,1600) * 1
    fig, ax = plt.subplots(3, figsize=(6.5, 4.5))
    ax[0].plot(t_range[:1300], cont_results_2['ref_traj'][0][:1300], linewidth=2, c='k', label='Reference')
    ax[0].plot(t_range[:1300], cont_results_1['S_controlled'][0][:1300] ,'--', linewidth=2, label='ESN')
    ax[0].plot(t_range[:1300], cont_results_2['S_controlled'][0][:1300] ,'--', linewidth=2, label='LSTM')

    ax[1].plot(t_range, cont_results_2['ref_traj'][1][:1600], linewidth=2, c='k')
    ax[1].plot(t_range, cont_results_1['S_controlled'][1] ,'--', linewidth=2, label='ESN')
    ax[1].plot(t_range, cont_results_2['S_controlled'][1] ,'--', linewidth=2, label='LSTM')

    ax[2].plot(t_range, cont_results_1['U'][0], '--', linewidth=2, label='ESN', c='tab:blue')
    ax[2].plot(t_range, cont_results_1['U'][1], '--', linewidth=2, label='ESN', c='tab:blue')
    ax[2].plot(t_range, cont_results_2['U'][0], '--', linewidth=2, label='ESN', c='tab:green')
    ax[2].plot(t_range, cont_results_2['U'][1], '--', linewidth=2, label='ESN', c='tab:green')

    ax[0].tick_params(axis="both", which="minor", labelsize=12)
    ax[1].tick_params(axis="both", which="minor", labelsize=12)
    ax[2].tick_params(axis="both", which="minor", labelsize=12)


    ax[0].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_ylabel('$x_2$', fontsize=12)
    ax[2].set_ylabel('$u$', fontsize=12)

    ax[2].set_xlabel("Time, $t$", fontsize=12)

    ax[0].legend(fontsize=10, loc="upper right")
    x_lims = ax[2].get_xlim()
    ax[0].set_xlim(x_lims)
    ax[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[2].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    plt.tight_layout()
    plt.savefig('two_tank_control.pdf')

if cont_results_1['sim'] == 'SpringMassControl':

    t_range = np.arange(0,1600) * 0.1
    fig, ax = plt.subplots(3, figsize=(6.5, 4.5))
    ax[0].plot(t_range[:1300], cont_results_1['ref_traj'][0][:1300], linewidth=2, c='k', label='Reference')
    ax[0].plot(t_range[:1300], cont_results_1['S_controlled'][0][:1300] ,'--', linewidth=2, label='ESN')
    ax[0].plot(t_range[:1300], cont_results_2['S_controlled'][0][:1300] ,'--', linewidth=2, label='LSTM')
    # ax[0].plot(t_range[:1300], lstm_results['S_controlled'][0][:1300] ,'--', linewidth=2, label='LSTM')


    ax[1].plot(t_range, cont_results_1['ref_traj'][1][:1600], linewidth=2, c='k')
    ax[1].plot(t_range, cont_results_1['S_controlled'][1] ,'--', linewidth=2, label='ESN')
    ax[1].plot(t_range, cont_results_2['S_controlled'][1] ,'--', linewidth=2, label='LSTM')
    # ax[1].plot(t_range, lstm_results['S_controlled'][1] ,'--', linewidth=2, label='LSTM')


    # ax[2].plot(t_range, cont_results['ref_traj'][2][:1600], linewidth=2, c='k')
    # ax[2].plot(t_range, cont_results['S_controlled'][2] ,'--', linewidth=2, label='ESN')
    # ax[2].plot(t_range, lstm_results['S_controlled'][2] ,'--', linewidth=2, label='LSTM')


    ax[2].plot(t_range, cont_results_1['U'][0], '--', linewidth=2, label='ESN')
    ax[2].plot(t_range, cont_results_2['U'][0], '--', linewidth=2, label='LSTM')
    # ax[2].plot(t_range, cont_results['U'][1], '--', linewidth=2, label='ESN')
    # ax[3].plot(t_range, lstm_results['U'][0], '--', linewidth=2, label='LSTM')

    ax[0].tick_params(axis="both", which="minor", labelsize=12)
    ax[1].tick_params(axis="both", which="minor", labelsize=12)
    ax[2].tick_params(axis="both", which="minor", labelsize=12)
    # ax[3].tick_params(axis="both", which="minor", labelsize=12)


    ax[0].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_ylabel('$x_2$', fontsize=12)
    ax[2].set_ylabel('$u$', fontsize=12)
    # ax[3].set_ylabel('$u$', fontsize=12)

    ax[0].set_ylim([-2.7, 2.7])
    ax[1].set_ylim([-2.7, 2.7])
    ax[2].set_ylim([-4, 4])

    ax[2].set_xlabel("Time, $t$", fontsize=12)

    ax[0].legend(fontsize=10, loc="upper right")
    x_lims = ax[2].get_xlim()
    ax[0].set_xlim(x_lims)
    ax[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    # ax[2].tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False)

    plt.tight_layout()
    plt.savefig('springmass_control.pdf')

if cont_results_1['sim'] == 'StirredTankControl':
    
    t_range = np.arange(0,1600) * 0.1
    fig, ax = plt.subplots(3, figsize=(6.5, 4.5))
    ax[0].plot(t_range[:1300], cont_results_1['ref_traj'][0][:1300], linewidth=2, c='k', label='Reference')
    ax[0].plot(t_range[:1300], cont_results_1['S_controlled'][0][:1300] ,'--', linewidth=2, label='ESN')
    ax[0].plot(t_range[:1300], cont_results_2['S_controlled'][0][:1300] ,'--', linewidth=2, label='LSTM')
    # ax[0].plot(t_range[:1300], lstm_results['S_controlled'][0][:1300] ,'--', linewidth=2, label='LSTM')


    ax[1].plot(t_range, cont_results_1['ref_traj'][1][:1600], linewidth=2, c='k')
    ax[1].plot(t_range, cont_results_1['S_controlled'][1] ,'--', linewidth=2, label='ESN')
    ax[1].plot(t_range, cont_results_2['S_controlled'][1] ,'--', linewidth=2, label='LSTM')
    # ax[1].plot(t_range, lstm_results['S_controlled'][1] ,'--', linewidth=2, label='LSTM')


    # ax[2].plot(t_range, cont_results['ref_traj'][2][:1600], linewidth=2, c='k')
    # ax[2].plot(t_range, cont_results['S_controlled'][2] ,'--', linewidth=2, label='ESN')
    # ax[2].plot(t_range, lstm_results['S_controlled'][2] ,'--', linewidth=2, label='LSTM')


    ax[2].plot(t_range, cont_results_1['U'][0], '--', linewidth=2, label='ESN')
    ax[2].plot(t_range, cont_results_2['U'][0], '--', linewidth=2, label='LSTM')
    # ax[2].plot(t_range, cont_results['U'][1], '--', linewidth=2, label='ESN')
    # ax[3].plot(t_range, lstm_results['U'][0], '--', linewidth=2, label='LSTM')

    ax[0].tick_params(axis="both", which="minor", labelsize=12)
    ax[1].tick_params(axis="both", which="minor", labelsize=12)
    ax[2].tick_params(axis="both", which="minor", labelsize=12)
    # ax[3].tick_params(axis="both", which="minor", labelsize=12)


    ax[0].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_ylabel('$x_2$', fontsize=12)
    ax[2].set_ylabel('$u$', fontsize=12)
    # ax[3].set_ylabel('$u$', fontsize=12)

    # ax[0].set_ylim([-2.7, 2.7])
    # ax[1].set_ylim([-2.7, 2.7])
    # ax[2].set_ylim([-4, 4])

    ax[2].set_xlabel("Time, $t$", fontsize=12)

    ax[0].legend(fontsize=10, loc="upper right")
    x_lims = ax[2].get_xlim()
    ax[0].set_xlim(x_lims)
    ax[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[2].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    plt.tight_layout()
    plt.savefig('stirred_tank_control.pdf')