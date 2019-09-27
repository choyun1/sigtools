#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""processing

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""

import warnings

from math import ceil, floor
# from numba import jit, int32, float64
import numpy as np
from numpy import cos, pi
from numpy.linalg import cholesky
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import hanning, resample, butter, lfilter

from sigtools.utils import *
from sigtools.sounds import *


def ramp_edges(sound, ramp_dur,
               ramp_func=lambda x: (1 + cos(pi*(1 - x)))/2):
    data = sound.data
    fs = sound.fs
    ramp_len = floor(fs*ramp_dur)
    ramp = ramp_func(np.linspace(0, 1, ramp_len)).reshape(-1, 1)
    data[:ramp_len]  = np.squeeze(ramp*data[:ramp_len].reshape(-1, len(data.shape)))
    data[-ramp_len:] = np.squeeze(np.flip(ramp)*data[-ramp_len:].reshape(-1, len(data.shape)))
    return Sound(data, fs)


def butter_bandpass_filter(sound, f_lo, f_hi, order=4):
    signal = sound.data
    fs = sound.fs
    f_nyq = fs/2
    lo = f_lo/f_nyq
    hi = f_hi/f_nyq
    b, a = butter(order, [lo, hi], btype="band")
    filtered_signal = lfilter(b, a, signal)
    rms_val = RMS(filtered_signal)
    return Sound(filtered_signal/rms_val, fs)


def equalize_fs(sound_list, option="downsample"):
    if option == "downsample":
        min_fs = min([sound.fs for sound in sound_list])
        new_sound_list = []
        for sound in sound_list:
            if sound.fs > min_fs:
                resample_len = floor(len(sound)*min_fs/sound.fs)
                resampled_sound_data = resample(sound.data, resample_len)
                new_sound_list.append(Sound(resampled_sound_data, min_fs))
            else:
                new_sound_list.append(sound)
        return new_sound_list
    elif option == "upsample":
        max_fs = max([sound.fs for sound in sound_list])
        new_sound_list = []
        for sound in sound_list:
            if sound.fs < max_fs:
                resample_len = floor(len(sound)*max_fs/sound.fs)
                resampled_sound_data = resample(sound.data, resample_len)
                new_sound_list.append(Sound(resampled_sound_data, max_fs))
            else:
                new_sound_list.append(sound)
        return new_sound_list
    else:
        raise ValueError("invalid option; must be 'downsample' or 'upsample'")


def truncate_sounds(sound_list):
    min_len = min([len(sound) for sound in sound_list])
    new_sound_list = [Sound(sound.data[:min_len], sound.fs) for sound in sound_list]
    return new_sound_list


def center_sounds(sound_list):
    max_len = max([len(sound) for sound in sound_list])
    new_sound_list = []
    for sound in sound_list:
        curr_len = len(sound)
        pad_len = max_len - curr_len
        sound_data = sound.data
        if pad_len > 0:
            if len(sound_data.shape) == 1:
                pad_dims = (0, pad_len)
            elif len(sound_data.shape) == 2:
                pad_dims = ((0, pad_len), (0, 0))
            min_val = np.min(np.abs(sound_data))
            sound_data = np.pad(sound_data, pad_dims,
                                mode="constant", constant_values=min_val)
            sound_data = np.roll(sound_data, pad_len//2, axis=0)
            new_sound_list.append(Sound(sound_data, sound.fs))
        else:
            new_sound_list.append(sound)
    return new_sound_list


def zeropad_sounds(sound_list):
    max_len = max([len(sound) for sound in sound_list])
    new_sound_list = []
    for sound in sound_list:
        curr_len = len(sound)
        pad_len = max_len - curr_len
        sound_data = sound.data
        if pad_len > 0:
            if len(sound_data.shape) == 1:
                pad_dims = (0, pad_len)
            elif len(sound_data.shape) == 2:
                pad_dims = ((0, pad_len), (0, 0))
            min_val = np.min(np.abs(sound_data))
            sound_data = np.pad(sound_data, pad_dims,
                                mode="constant", constant_values=min_val)
            new_sound_list.append(Sound(sound_data, sound.fs))
        else:
            new_sound_list.append(sound)
    return new_sound_list


def normalize_rms(sound_list):
    new_sound_list = [Sound(sound.data/RMS(sound.data), sound.fs) for sound in sound_list]
    return new_sound_list


def zero_mean(sound_list):
    new_sound_list = [Sound(sound.data - np.mean(sound.data, axis=0), sound.fs) for sound in sound_list]
    return new_sound_list


def concat_sounds(sound_list):
    fs = sound_list[0].fs
    concat_data = np.concatenate([sound.data for sound in sound_list])
    return Sound(concat_data, fs)


# TODO: Correlated sounds not quite working correctly
# @jit(float64[:, :](int32, int32, float64, float64, float64), nopython=True)
# def make_covariance_matrix(n_channels, n_windows, fcorr, tcorr, mod_depth):
#     """make_covariance_matrix
#
#     Parameters
#     ----------
#     n_channels : int
#         Number of frequency channels
#     n_windows : int
#         Number of time windows
#     fcorr : float
#         Correlation between frequency channels; in units of subbands
#     tcorr : float
#         Correlation between time windows
#     mod_depth : float
#         Modulation depth
#
#     Returns
#     -------
#     out : 2-D array of floats
#     """
#     fcorr_arr = np.exp(-fcorr*np.arange(n_channels))
#     tcorr_arr = np.exp(-tcorr*np.arange(n_windows))
#
#     # Correlation matrix size must match the number of grid elements in TF
#     # representation
#     n_grid_elements = n_channels*n_windows
#     C = np.zeros((n_grid_elements, n_grid_elements))
#     for j in range(n_grid_elements):
#         for k in range(j, n_grid_elements):
#             sub_j = j % n_channels
#             sub_k = k % n_channels
#             time_j = j // n_channels
#             time_k = k // n_channels
#
#             curr_fcorr = fcorr_arr[abs(sub_j - sub_k)]
#             curr_tcorr = tcorr_arr[abs(time_j - time_k)]
#             C[j, k] = mod_depth*curr_fcorr*curr_tcorr
#             C[k, j] = C[j, k]
#     return C
#
#
# def make_correlated_TF_grid(n_channels, n_windows, C, R):
#     T = cholesky(C)
#     return np.matmul(T, R).reshape(n_channels, n_windows)
#
#
# @jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :], int32, int32),
#      nopython=True)
# def make_modified_subbands(energy_grid, subbands, windows,
#                            n_channels, n_windows):
#     """make_modified_subbands
#
#     Parameters
#     ----------
#     energy_grid : Sound object
#         Input sound that will be transformed using the correlated TF grid.
#     subbands : Numpy array of floats
#     win_arr : Numpy array of floats
#
#     Returns
#     -------
#     out : Numpy array of floats
#     """
#     mod_subbands = np.zeros(subbands.shape)
#     for curr_sub_i in range(n_channels):
#         curr_subband = subbands[:, curr_sub_i]
#         mod_subband_accum = np.zeros(curr_subband.shape)
#         for curr_win_i in range(n_windows):
#             curr_sub_win = curr_subband*windows[:, curr_win_i]
#             mod_subband_accum +=\
#                 np.sqrt(np.mean(np.square(curr_sub_win))) *\
#                 energy_grid[curr_sub_i, curr_win_i]*curr_sub_win
#         mod_subbands[:, curr_sub_i] = mod_subband_accum
#     return mod_subbands
#
#
# def make_correlated_sound(sound, n_channels=16, f_lo=0, f_hi=44100,
#                           win_dur=20e-3, win_overlap=0.5, win_func=hanning,
#                           fcorr=0.065, tcorr=0.109, mod_depth=0.5):
#     """make_correlated_sound
#
#     Parameters
#     ----------
#     input_sound : Sound object
#         Input sound that will be transformed using the correlated TF grid.
#
#     Returns
#     -------
#     out : Sound object
#     """
#     from sigtools.representations import Subbands, ModifiedSubbands
#     fs = sound.fs
#     win_overlap = 0.5
#     win_len = floor(fs*win_dur)
#     adv_len = floor(win_len*(1 - win_overlap))
#     n_wins = floor((len(sound) - win_len)/adv_len)
#     window = win_func(win_len)
#
#     subbands = Subbands(sound, n_channels, f_lo, f_hi)
#     subbands_arr = np.array([subbands.subbands[i].data for i in range(n_channels)]).T
#     windows_arr = np.zeros((len(sound), n_wins))
#     for i in range(n_wins):
#         curr_beg_idx = i*adv_len
#         curr_end_idx = curr_beg_idx + win_len
#         windows_arr[curr_beg_idx:curr_end_idx, i] = window
#
#     C = make_covariance_matrix(n_channels, n_wins, fcorr, tcorr, mod_depth)
#     R = np.random.randn(n_channels*n_wins, 1)
#     corred_grid = make_correlated_TF_grid(n_channels, n_wins, C, R)
#     cell_mean = -0.75
#     log_const = 1e-12
#     energy_grid = np.power(10, corred_grid + cell_mean - log_const)
#
#     modified_subbands_arr = make_modified_subbands(energy_grid, subbands_arr, windows_arr,
#                                                    n_channels, n_wins)
#     modified_subbands_list = [Sound(modified_subbands_arr[:, i], fs) for i in range(n_channels)]
#     modified_subbands = ModifiedSubbands(modified_subbands_list,
#                                          subbands.filterbank, subbands.CFs, subbands.fs)
#     corred_sound = modified_subbands.to_Sound()
#     return corred_sound
