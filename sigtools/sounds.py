#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""sounds

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""

import warnings

from math import ceil, floor
import numpy as np
from numpy import pi, sin, cos, tan, arctan
from numpy.fft import rfft, irfft
from scipy.signal import hilbert, resample
from scipy.signal import convolve as sp_convolve
from scipy.io.wavfile import  read as wavread
from scipy.io.wavfile import write as wavwrite

from sigtools.utils import *


class Sound:
    def __init__(self, data, fs):
        self.data = np.squeeze(data)
        self.fs = fs

    def __len__(self):
        return self.data.shape[0]

    def __neg__(self):
        return Sound(-self.data, self.fs)

    def __add__(self, other):
        if isinstance(other, Sound):
            new_data = self.data + other.data
            return Sound(new_data, self.fs)
        elif isinstance(other, int) or isinstance(other, float):
            return Sound(10**(other/10)*self.data, self.fs)
        else:
            raise RuntimeError("invalid addend for Sound; must be a Sound object or a real number")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, Sound):
            self_data = self.data
            self_fs = self.fs
            other_data = other.data
            return Sound(self_data*other_data, self_fs)
        elif isinstance(other, int) or isinstance(other, float):
            self_data = self.data
            self_fs = self.fs
            return Sound(self_data*other, self_fs)
        else:
            raise RuntimeError("invalid multiplier for Sound; must be a Sound object or a real number")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Sound):
            self_data = self.data
            self_fs = self.fs
            other_data = other.data
            return Sound(self_data/other_data, self_fs)
        elif isinstance(other, int) or isinstance(other, float):
            self_data = self.data
            self_fs = self.fs
            return Sound(self_data/other, self_fs)
        else:
            raise RuntimeError("invalid multiplier for Sound; must be a Sound object or a real number")

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def make_binaural(self):
        if len(self.data.shape) == 2:
            raise RuntimeError("sound is already binaural")
        else:
            new_data = np.tile(self.data.reshape(-1, 1), (1, 2))
            return Sound(new_data, self.fs)

    def convolve(self, other):
        self_data = self.data
        self_fs = self.fs
        other_data = other.data
        if len(self.data.shape) == 1:
            new_data = sp_convolve(self_data, other_data)
        elif len(self.data.shape) == 2:
            new_data = np.array( [sp_convolve(self_data[:, i], other_data[:, i]) for i in range(2)] ).T
        return Sound(new_data, self.fs)

    def save(self, path):
        max = np.max(np.abs(self.data))
        wavwrite(path, self.fs, self.data/max)

    def extract_envelope(self):
        data = self.data
        fs = self.fs
        env = np.abs(hilbert(data, axis=0))
        return Sound(env, fs)


class Periodic(Sound):
    def __init__(self, data, fs, f0, phase):
        Sound.__init__(self, data, fs)
        self.f0 = f0


class PureTone(Periodic):
    def __init__(self, sig_dur, fs, f0, phase=0):
        sig_len = floor(fs*sig_dur)
        t = np.linspace(0, sig_dur, sig_len)
        pure_tone_data = cos(2*pi*f0*t + phase)
        rms_val = RMS(pure_tone_data)
        Periodic.__init__(self, pure_tone_data/rms_val, fs, f0, phase)


class SquareWave(Periodic):
    def __init__(self, sig_dur, fs, f0, phase=0):
        sig_len = floor(fs*sig_dur)
        t = np.linspace(0, sig_dur, sig_len)
        square_data = np.sign(sin(2*pi*f0*t + phase))
        rms_val = RMS(square_data)
        Periodic.__init__(self, square_data/rms_val, fs, f0, phase)


class SawtoothWave(Periodic):
    def __init__(self, sig_dur, fs, f0, phase=0):
        sig_len = floor(fs*sig_dur)
        t = np.linspace(0, sig_dur, sig_len)
        sawtooth_data = -arctan(1/tan(pi*f0*t + phase))
        rms_val = RMS(sawtooth_data)
        Periodic.__init__(self, sawtooth_data/rms_val, fs, f0, phase)


class PulseTrain(Periodic):
    def __init__(self, sig_dur, fs, f0, phase=0):
        sig_len = int(fs*sig_dur)
        T_samples = fs/f0
        phase_samples = round(T_samples*phase/2*pi)
        pulse_idxs = np.arange(0, sig_len, fs/f0, dtype=int)
        pulse_train_data = np.zeros(sig_len)
        pulse_train_data[pulse_idxs] = 0.5
        pulse_train_data[pulse_idxs + 1] = -0.5
        pulse_train_data = np.roll(pulse_train_data, phase_samples)
        rms_val = RMS(pulse_train_data)
        Periodic.__init__(self, pulse_train_data/rms_val, fs, f0, phase)


class HarmonicComplex(Periodic):
    def __init__(self, sig_dur, fs, f0, harmonics, amplitudes, phases):
        sig_len = floor(fs*sig_dur)
        f_nyq = fs/2
        harmonics = harmonics.astype(int) # force integer harmonics
        f = harmonics*f0
        f = f[np.where(f < f_nyq)[0]]
        sound_sum = Silence(1, 1)
        for i in range(len(harmonics)):
            curr_sound = amplitudes[i]*PureTone(sig_dur, fs, f[i], phases[i])
            sound_sum = sum([sound_sum, curr_sound])
        harm_complex_data = sound_sum.data
        rms_val = RMS(harm_complex_data)
        Periodic.__init__(self, harm_complex_data/rms_val, fs, f0, phases)
        self.harmonics = harmonics
        self.amplitudes = amplitudes


class SchroederPhase(Periodic):
    def __init__(self, sig_dur, fs, f0, n_harmonics=100):
        sig_len = floor(fs*sig_dur)
        A = 1/n_harmonics
        t = np.linspace(0, sig_dur, sig_len)
        schroeder_data = np.zeros(t.size)
        for k in range(n_harmonics):
            schroeder_data += A*cos(2*pi*(k + 1)*f0*t + pi*(k + 1)**2/n_harmonics)
        rms_val = RMS(schroeder_data)
        Sound.__init__(self, schroeder_data/rms_val, fs)
        self.f0 = f0


class Chirp(Sound):
    def __init__(self, data, fs, f0, f1):
        Sound.__init__(self, data, fs)
        self.f0 = f0
        self.f1 = f1


class LinearChirp(Chirp):
    def __init__(self, sig_dur, fs, f0, f1, phase=0):
        sig_len = floor(fs*sig_dur)
        t = np.linspace(0, sig_dur, sig_len)
        k = (f1 - f0)/sig_dur
        lin_chirp_data = sin(2*pi*(f0*t + k/2*t**2) + phase)
        rms_val = RMS(lin_chirp_data)
        Chirp.__init__(self, lin_chirp_data/rms_val, fs, f0, f1)
        self.k = k


class ExponentialChirp(Chirp):
    def __init__(self, sig_dur, fs, f0, f1, phase=0):
        sig_len = floor(fs*sig_dur)
        t = np.linspace(0, sig_dur, sig_len)
        k = (f1/f0)**(1/sig_dur)
        exp_chirp_data = sin(2*pi*f0*(k**t - 1)/np.log(k) + phase)
        rms_val = RMS(exp_chirp_data)
        Chirp.__init__(self, exp_chirp_data/rms_val, fs, f0, f1)
        self.k = k


class Noise(Sound):
    def __init__(self, data, fs):
        Sound.__init__(self, data, fs)


class GaussianNoise(Noise):
    def __init__(self, sig_dur, fs, f_lo=0, f_hi=44100, spec_tilt=0, custom_spec_env=None):
        sig_len = floor(fs*sig_dur)
        f_nyq = fs/2
        if f_hi > f_nyq:
            f_hi = f_nyq
        spec_len = sig_len//2 + 1
        if custom_spec_env is not None:
            resampled_spec_env = resample(custom_spec_env, spec_len)
            spec_env = np.power(10, resampled_spec_env/10)
        else:
            freqs = np.linspace(1, f_nyq, spec_len)
            spec_env = np.power(freqs, spec_tilt/10)

        init_noise = np.random.randn(sig_len)
        noise_spec = rfft(init_noise)
        idx_lo =  ceil(spec_len*f_lo/f_nyq)
        idx_hi = floor(spec_len*f_hi/f_nyq)
        noise_spec[:idx_lo] = 0.
        noise_spec[idx_hi:] = 0.
        noise_spec *= spec_env

        noise_data = irfft(noise_spec, sig_len)
        noise_data -= np.mean(noise_data)
        rms_val = RMS(noise_data)

        Noise.__init__(self, noise_data/rms_val, fs)
        self.spec_tilt = spec_tilt
        self.f_lo = f_lo
        self.f_hi = f_hi


class TORC(Noise):
    def __init__(self):
        raise NotImplementedError


class CorrelatedNoise(Noise):
    def __init__(self, sig_dur, fs, f_lo=0, f_hi=44100, spec_tilt=0, corr=1):
        sig_len = floor(fs*sig_dur)
        alpha = np.sqrt((corr + 1)/2)
        beta  = np.sqrt(1 - alpha**2)

        noise_data = np.zeros((sig_len, 2))
        noise1 = GaussianNoise(sig_dur, fs, f_lo, f_hi, spec_tilt)
        noise2 = GaussianNoise(sig_dur, fs, f_lo, f_hi, spec_tilt)
        noise_data[:, 0] = alpha*noise1.data + beta*noise2.data
        noise_data[:, 1] = alpha*noise1.data - beta*noise2.data
        Noise.__init__(self, noise_data, fs)
        self.spec_tilt = spec_tilt
        self.corr = corr


class ImpulseResponse(Sound):
    def __init__(self, data, fs):
        Sound.__init__(self, data, fs)


class SynthIR(ImpulseResponse):
    def __init__(self, synth_DRR, synth_RT60, dB_thresh,
                 fs, f_lo=0, f_hi=44100,
                 n_channels=16, env_type="exponential"):
        from sigtools.representations import Subbands, ModifiedSubbands
        model_dir = "../reverb_data/"
        model_DRR   = np.load(model_dir + "fit_DRR.npy")
        model_RT60  = np.load(model_dir + "fit_RT60.npy")
        model_freqs = np.load(model_dir + "fit_freqs.npy")

        # Compute the frequencies over which synth parameters will be computed
        ERB_lo = freq_to_ERB(f_lo)
        ERB_hi = freq_to_ERB(f_hi)
        interp_ERB_domain  = np.linspace(ERB_lo, ERB_hi, n_channels)
        interp_freq_domain = ERB_to_freq(interp_ERB_domain)

        # Compute synthetic DRR for each frequency channel
        temp_a1 = model_DRR[:, 0]*np.log10(synth_RT60) + model_DRR[:, 1]
        temp_a2 = temp_a1 - np.median(temp_a1[1:-1]) + synth_DRR
        synth_a = np.interp(interp_freq_domain, model_freqs, temp_a2)

        # Compute synthetic RT60 for each frequency channel
        temp_b1 = np.power(10, model_RT60[:, 0]*np.log10(synth_RT60) + model_RT60[:, 1])
        temp_b2 = 60./temp_b1
        synth_b = np.interp(interp_freq_domain, model_freqs, temp_b2)

        # Calculate maximum possible required time [sec] given the dB threshold
        # assuming linear decay in log freq scale (i.e. exponential amplitude env)
        max_t_s = np.max(np.abs(dB_thresh)/synth_b)

        # Generate the Gaussian noise that will be used as the basis for synthesis
        gnoise = GaussianNoise(max_t_s, fs, f_lo, f_hi)

        # Calculate time points and generate a filterbank
        synth_IR_max_len = len(gnoise)
        time_pts = np.linspace(0, max_t_s, synth_IR_max_len)
        gnoise_subbands = Subbands(gnoise, n_channels, f_lo, f_hi)

        # Create holder arrays for the IR subbands and threshold crossing indexes
        # for later truncation
        synth_IR_subbands = np.zeros((len(gnoise_subbands.subbands[0]),
                                      n_channels))
        thresh_cross_idxs = np.zeros(n_channels, dtype=int)
        for curr_channel in range(n_channels):
            curr_a = synth_a[curr_channel]
            curr_b = synth_b[curr_channel]
            curr_subband = gnoise_subbands.subbands[curr_channel].data
            curr_subband = curr_subband.reshape(-1, len(curr_subband.shape))

            # Calculate the time [sec] and array index at which the linear
            # envelope passes below dB_thresh.
            # NOTE: dB_thresh is expected to be negative!
            curr_thresh_idx = np.where(curr_b*time_pts >= np.abs(dB_thresh))[0][0]
            thresh_cross_idxs[curr_channel] = curr_thresh_idx
            curr_thresh_t_s = curr_thresh_idx/fs

            # Compute the area under the exponential envelope in units of [dB*s]
            # Defined by a triangle - bounded on the bottom by curr_thresh_t_s
            dB_below_zero_at_thresh = np.abs(curr_a - np.abs(dB_thresh))
            A_exp = (curr_a + dB_below_zero_at_thresh)*curr_thresh_t_s/2

            if env_type == "exponential":
                curr_env_dB = curr_a - curr_b*time_pts
            elif env_type == "time_reversed":
                curr_env_dB = curr_a - curr_b*time_pts
                curr_env_dB = curr_env_dB[::-1]
            elif env_type == "lin_match_beg":
                # Create a linear envelope (logarithmic in dB scale) that
                # matches the area (i.e. energy) of the exponential envelope
                # while also matching the starting DRR (curr_a). Coefficients
                # were determined algebraically.
                a_prime = curr_a + dB_below_zero_at_thresh
                b_prime = -A_exp/(20/np.log(10) + a_prime*np.power(10, a_prime/20)\
                                                 /(1 - np.power(10, a_prime/20)))
                curr_time_idx = int(b_prime/max_t_s*time_pts.size)

                thresh_cross_idxs[curr_channel] = curr_time_idx
                c1 = (1 - np.power(10, a_prime/20))/b_prime
                c2 = np.power(10, a_prime/20)
                new_time_pts = time_pts[:curr_time_idx]
                log_env = 20*np.log10(c1*new_time_pts + c2) - dB_below_zero_at_thresh
                end_pad = log_env[-1]*np.ones(time_pts.size - new_time_pts.size)
                curr_env_dB = np.concatenate((log_env, end_pad))
            elif env_type == "lin_match_end":
                # Creates a linear envelope while matching the threshold crossing
                b_prime = curr_thresh_t_s
                a_prime = fsolve(
                    lambda x: A_exp + b_prime*(20/np.log(10) \
                                           + x*np.power(10, x/20)/(1 - np.power(10, x/20)) ),
                    curr_a)
                curr_time_idx = curr_thresh_idx

                thresh_cross_idxs[curr_channel] = curr_time_idx
                c1 = (1 - np.power(10, a_prime/20))/b_prime
                c2 = np.power(10, a_prime/20)
                new_time_pts = time_pts[:curr_time_idx]
                log_env = 20*np.log10(c1*new_time_pts + c2) - dB_below_zero_at_thresh
                end_pad = log_env[-1]*np.ones(time_pts.size - new_time_pts.size)
                curr_env_dB = np.concatenate((log_env, end_pad))
            else:
                raise ValueError("invalid env_type")
            curr_env_amp = np.power(10, curr_env_dB/20).reshape(-1, 1)
            synth_IR_subbands[:, curr_channel] = np.squeeze(curr_env_amp*curr_subband)
        max_thresh_cross_idx = max(thresh_cross_idxs)
        synth_IR_subbands = synth_IR_subbands[:max_thresh_cross_idx]
        truncated_filterbank = gnoise_subbands.filterbank[:max_thresh_cross_idx//2 + 1]
        synth_IR_subbands_list = [Sound(synth_IR_subbands[:, i], fs)
                                  for i in range(n_channels)]
        # Find the index at which to truncate (less than dB_thresh for all
        # subbands) and return the normalized synthetic IR as a Sound object
        synth_IR = ModifiedSubbands(synth_IR_subbands_list,
                                    truncated_filterbank,
                                    gnoise_subbands.CFs, fs).to_Sound()
        synth_IR_data = synth_IR.data
        synth_IR_data = synth_IR_data[:floor(0.99*synth_IR_data.size)]
        rms_val = RMS(synth_IR_data)
        ImpulseResponse.__init__(self, synth_IR_data/rms_val, fs)


class SimpleBIR(ImpulseResponse):
    def __init__(self, fs, ITD=0., ILD=0.):
        IR_len = floor(fs*abs(ITD))
        if IR_len < 1:
            warnings.warn("input ITD is unresolvable given the sampling rate; setting ITD to 0")
            IR_len = 1
        BIR = np.zeros((IR_len, 2))
        if ITD > 0:
            BIR[-1, 0] = 1
            BIR[ 0, 1] = 1
        else:
            BIR[ 0, 0] = 1
            BIR[-1, 1] = 1
        BIR[:, 1] = 10**(ILD/20)*BIR[:, 1]
        rms_val = RMS(BIR)
        ImpulseResponse.__init__(self, BIR/rms_val, fs)


class HRIR(ImpulseResponse):
    def __init__(self, data, fs):
        ImpulseResponse.__init__(self, data, fs)


class SoundLoader(Sound):
    def __init__(self, path):
        fs, y = wavread(path)
        y = y.astype(np.float32)
        rms_val = RMS(y)
        Sound.__init__(self, y/rms_val, fs)


class Silence(Sound):
    def __init__(self, sig_dur, fs):
        sig_len = floor(fs*sig_dur)
        silence_data = np.zeros(sig_len)
        Sound.__init__(self, silence_data, fs)
