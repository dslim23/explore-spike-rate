import streamlit as st
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(data_file):
    data_mat = sio.loadmat(data_file)
    spike_data = data_mat['all_maps']
    trial_idx = (data_mat['trialStarts'].flatten() - 1).tolist()
    kilosort_neuron_id = data_mat['cells_to_plot']
    return spike_data, trial_idx, kilosort_neuron_id


def get_block_idx(block_num, trial_idx):
    return trial_idx[block_num - 1], trial_idx[block_num]


def filter_spike_data(spike_data,
                      trial_idx,
                      neuron=None,
                      block=None,
                      trial=None):

    filtered = spike_data[neuron - 1:neuron] if neuron else spike_data

    filtered = spike_data[:, trial - 1:trial] if trial else filtered

    if block:
        start, end = get_block_idx(block, trial_idx)

        filtered = filtered[:, start:end]

    return filtered


def plot_one(data, summary=False, dividers=None,x_tick_interval = 20, y_tick_interval=40, xlabel='Position'):
    data = np.vstack(data).squeeze()

    if data.ndim == 1:
        data = data[np.newaxis, ]

    trial_n, distance_n = data.shape
    min_ = min(distance_n, trial_n)
    x_size = trial_n / 30

    if summary:
        x_size = trial_n / 2
    fig, axes = plt.subplots(figsize=(10, x_size))
    sns.heatmap(data, ax=axes)
    axes.set(xlabel=xlabel, ylabel='Trial')

    x_idx = list(range(0, distance_n, x_tick_interval)) + [distance_n] if x_tick_interval else []
    axes.set_xticks(x_idx)
    axes.set_xticklabels(x_idx)

    y_idx = list(range(y_tick_interval, trial_n, y_tick_interval)) if y_tick_interval else []

    axes.set_yticks(y_idx)
    axes.set_yticklabels(y_idx)
    if dividers:
        dividers = dividers[1:-1]
        axes.hlines(dividers,
                    *axes.get_xlim(),
                    colors='white',
                    linestyles='dotted')

    st.pyplot(fig)
    return trial_n, fig, dividers, distance_n
st.title('Firing Rate Visualizer')
data_file = st.sidebar.file_uploader('Upload .mat file', type='.mat')
if data_file:
    spike_data, trial_idx, kilosort_neuron_id = load_data(data_file)
    block_n = len(trial_idx)
    neuron_n, trial_n, timebins_per_trial = spike_data.shape
    trial_idx.append(trial_n)

    neurons_to_display = st.multiselect('Neurons',
                                        range(1, neuron_n + 1),
                                        default=range(1, neuron_n + 1))
    blocks = st.multiselect('Block', range(1, block_n + 1), default=1)

    show_position_chart = st.sidebar.checkbox('Show position chart', value=True)
    show_distance_chart = st.sidebar.checkbox('Show distance chart', value=True)
    show_mean_std = st.sidebar.checkbox('Show mean and std per block', value=False)

    run = st.button('Run')

    if run and neurons_to_display and blocks:
        neurons_to_display_n = len(neurons_to_display)
        blocks_n = len(blocks)
        chart_n = neurons_to_display_n * blocks_n

        fig, axes = plt.subplots(len(neurons_to_display) * len(blocks))

        pbar = st.progress(0)
        for i, neuron_ in enumerate(neurons_to_display):
            st.header(f'Neuron {neuron_}')
            filtered_ = []
            std_ = []
            mean_ = []
            dividers = [0]
            for j, block_ in enumerate(blocks):

                filtered = filter_spike_data(spike_data,
                                            trial_idx,
                                            neuron=neuron_,
                                            block=block_)
                filtered = filtered.squeeze()
                if show_mean_std:
                    std = np.std(filtered, axis=0)[np.newaxis, ]
                    mean = np.mean(filtered, axis=0)[np.newaxis, ]

                    std_.append(std)
                    mean_.append(mean)
                filtered_.append(filtered)
                dividers.append(dividers[-1] + filtered.shape[0])
            if show_position_chart:
                st.header(f'Position')
                trial_n, fig, dividers, distance_n = plot_one(
                    filtered_, False, dividers)
            if show_distance_chart:
                st.header(f'Distance')
                flattened = [f.flatten() for f in filtered_]
                plot_one(flattened, True, x_tick_interval = None, y_tick_interval=None, xlabel='Distance')
            #STD
            if show_mean_std:
                st.subheader('Mean per block')
                trial_n, fig, dividers, distance_n = plot_one(
                    mean_, True, dividers)
                st.subheader('Standard deviation per block')

                trial_n, fig, dividers, distance_n = plot_one(std_, True, dividers)
            pbar.progress((i + 1) / neurons_to_display_n)
