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


def compare_relative_db(sound_list):
    if len(sound_list) == 1:
        return [0]
    else:
        ref_rms = RMS(sound_list[0].data)
        return [20*np.log10(RMS(snd.data)/ref_rms) for snd in sound_list]
