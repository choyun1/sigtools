#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""representations

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""

from abc import ABC, abstractmethod

from math import ceil, floor
import numpy as np
from numpy import pi, cos
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq, fft2
# from scipy.fft import fft, ifft
from scipy.signal import hanning, correlate, resample

from sigtools.utils import *
from sigtools.sounds import *

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# import IPython.display as ipd


DEFAULT_CMAP = cm.magma


class Representation(ABC):
    def __init__(self, sound):
        self.sound = sound
        super().__init__()

    @abstractmethod
    def display(self):
        pass


class AudioControl(Representation):
    def display(self):
        sound = self.sound
        data = sound.data
        fs = sound.fs
        # ipd.display(ipd.Audio(data=data.T, rate=fs))


class Waveform(Representation):
    def display(self, ax, n_xticks=6):
        sound = self.sound
        data = sound.data
        fs = sound.fs

        amp_max = 1.1*np.max(np.abs(data))
        t = np.linspace(0, len(sound)/fs, len(sound))

        ax.plot(t, data, alpha=0.5)
        ax.hlines(0, t[0], t[-1], color="k", linestyle="-", alpha=0.25)
        ax.set_title("Waveform")
        xtick_locs = np.linspace(t[0], t[-1], n_xticks, endpoint=True)
        xtick_strs = ["{:.1f}".format(t_i) for t_i in xtick_locs]
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_strs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim((t[0], t[-1]))
        ax.set_ylim((-amp_max, amp_max))
        ax.grid(linestyle=":")


class Subbands(Representation):
    def __init__(self, sound, n_channels=8, f_lo=0, f_hi=44100, fscale="ERB",
                 filter_shape=lambda x: (1 + cos(x))/2):
        Representation.__init__(self, sound)
        sig_len = len(sound)
        spec_len = sig_len//2 + 1
        data = sound.data
        fs = sound.fs
        f_nyq = fs/2
        if f_hi > f_nyq:
            f_hi = f_nyq - 0.1

        if fscale == "linear":
            raise NotImplementedError

        elif fscale == "ERB":
            ERB_lo, ERB_hi = freq_to_ERB( np.array([f_lo, f_hi]) )
            ERB_1D = np.linspace(ERB_lo, ERB_hi, n_channels)
            f = np.linspace(0, f_nyq, spec_len, endpoint=True)
            ERB_cutoff_pairs = np.array([[ERB_1D[i], ERB_1D[i + 2]] for i in range(n_channels - 2)])
            f_cutoff_pairs = ERB_to_freq(ERB_cutoff_pairs)

            filterbank = np.zeros((spec_len, n_channels))
            CFs = np.zeros(n_channels)
            for curr_channel in range(1, n_channels - 1):
                curr_f_lo, curr_f_hi = f_cutoff_pairs[curr_channel - 1]
                curr_idx_lo = np.argmax(f > curr_f_lo)
                curr_idx_hi = np.argmin(f < curr_f_hi)
                curr_mean_ERB = np.mean(ERB_cutoff_pairs[curr_channel - 1, :])
                curr_ERB_bandwidth = ERB_cutoff_pairs[curr_channel - 1, 1] \
                                   - ERB_cutoff_pairs[curr_channel - 1, 0]
                curr_ERBs = freq_to_ERB(f[curr_idx_lo:curr_idx_hi])
                normalized_domain = 2*(curr_ERBs - curr_mean_ERB)/curr_ERB_bandwidth
                curr_filter = filter_shape(pi*normalized_domain)
                filterbank[curr_idx_lo:curr_idx_hi, curr_channel] = curr_filter
                CFs[curr_channel] = ERB_to_freq(curr_mean_ERB)
            # Make the low-pass and high-pass filters at the edges
            lopass_f = f_cutoff_pairs[0, 1]
            lopass_idx = np.argmax(f > lopass_f)
            lopass_domain = np.linspace(0, pi, len(range(lopass_idx)))
            lopass_filter = filter_shape(lopass_domain)
            filterbank[:lopass_idx, 0] = lopass_filter
            CFs[0] = f[0]

            hipass_f = f_cutoff_pairs[-1, 0]
            hipass_idx = np.argmin(f < hipass_f)
            hipass_domain = np.linspace(-pi, 0, len(range(hipass_idx, spec_len)))
            hipass_filter = filter_shape(hipass_domain)
            filterbank[hipass_idx:, -1] = hipass_filter
            CFs[-1] = f[-1]

            subbands = []
            rfft_data = rfft(data, axis=0).reshape(-1, len(data.shape))
            for i in range(n_channels):
                curr_filt = filterbank[:, i].reshape(-1, 1)
                curr_filt_data = curr_filt*rfft_data
                curr_irfft = irfft(curr_filt_data, sig_len, axis=0)
                subbands.append(Sound(curr_irfft, fs))
            self.subbands = subbands
            self.filterbank = filterbank
            self.CFs = CFs
            self.fs = fs
        else:
            raise ValueError("fscale must be 'linear' or 'ERB'")

    def __len__(self):
        return len(self.subbands)

    def __mul__(self, other):
        if isinstance(other, Subbands):
            if len(self) != len(other):
                raise RuntimeError("number of channels in subbands must match")
            self_subbands = self.subbands
            other_subbands = other.subbands
            n_channels = len(self)
            modified_subbands = [self_subbands[i]*other_subbands[i] for i in range(n_channels)]
        else:
            raise RuntimeError("cannot multiply a subband with something else")
        return ModifiedSubbands(modified_subbands, self.filterbank, self.CFs, self.fs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Subbands):
            if len(self) != len(other):
                raise RuntimeError("number of channels in subbands must match")
            self_subbands = self.subbands
            other_subbands = other.subbands
            n_channels = len(self)
            modified_subbands = [self_subbands[i]/other_subbands[i] for i in range(n_channels)]
        else:
            raise RuntimeError("cannot multiply a subband with something else")
        return ModifiedSubbands(modified_subbands, self.filterbank, self.CFs, self.fs)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def extract_envelopes(self):
        modified_subbands = [sound.extract_envelope() for sound in self.subbands]
        return ModifiedSubbands(modified_subbands, self.filterbank, self.CFs, self.fs)

    def to_Sound(self):
        subbands = self.subbands
        filterbank = self.filterbank
        fs = self.fs
        n_channels = len(subbands)
        sig_len = len(subbands[0])

        filtered_subbands = []
        for i in range(n_channels):
            rfft_subband = rfft(subbands[i].data, axis=0).reshape(-1, len(subbands[i].data.shape))
            curr_filt = filterbank[:, i].reshape(-1, 1)
            curr_filt_subband = curr_filt*rfft_subband
            curr_irfft = irfft(curr_filt_subband, sig_len, axis=0)
            filtered_subbands.append(curr_irfft)
        filtered_subbands = np.array(filtered_subbands)
        summed_data = np.sum(filtered_subbands, axis=0)
        rms_val = RMS(summed_data)
        return Sound(summed_data/rms_val, fs)

    def display(self, axes):
        subbands = self.subbands
        CFs = self.CFs
        n_channels = len(self)
        if len(axes) != n_channels:
            raise RuntimeError("the number of axes must match the number of channels")

        for i in range(n_channels):
            curr_subband = subbands[i]
            ax_idx = list(reversed(range(n_channels)))[i]
            Waveform(curr_subband).display(axes[ax_idx])
            axes[ax_idx].set_title("")
            axes[ax_idx].set_xlabel("")
            axes[ax_idx].set_ylabel("Chan. {:d}\nCF: {:.0f} Hz".format(i + 1, CFs[i]))
            if i != n_channels - 1:
                axes[ax_idx].set_xticklabels([])
            axes[ax_idx].set_yticklabels([])
        axes[0].set_title("Subbands")
        axes[ax_idx].set_xlabel("Time [s]")



class ModifiedSubbands(Subbands):
    def __init__(self, modified_subbands, filterbank, CFs, fs):
        self.subbands = modified_subbands
        self.filterbank = filterbank
        self.CFs = CFs
        self.fs = fs


class MagnitudeSpectrum(Representation):
    def __init__(self, sound):
        Representation.__init__(self, sound)
        data = sound.data
        fs = sound.fs
        f_nyq = fs/2
        spec_len = len(sound)//2 + 1
        self.fft = fft(data, axis=0)
        self.log_mag_spect = 10*np.log10(np.abs(rfft(data, axis=0)))
        self.f = np.linspace(0, f_nyq, spec_len)

    def __mul__(self, other):
        if isinstance(other, MagnitudeSpectrum):
            self_log_mag_spect = self.log_mag_spect
            other_log_mag_spect = other.log_mag_spect
            if self_log_mag_spect.shape != other_log_mag_spect.shape:
                other_log_mag_spect = resample(other_log_mag_spect, self_log_mag_spect.shape[0])
            modified_log_mag_spect = self_log_mag_spect*other_log_mag_spect
            return ModifiedMagSpectrum(modified_log_mag_spect, self.f)
        else:
            raise RuntimeError("invalid multiplicand; must be another magnitude spectrum")

    def extract_envelope(self, quefrency_cutoff=0.1):
        log_mag_spect = self.log_mag_spect
        f = self.f
        f_nyq = f[-1]
        n_samples = log_mag_spect.shape[0]
        sample_cutoff = floor(quefrency_cutoff*n_samples)
        cepstrum = fft(log_mag_spect)
        cepstrum_window = np.ones(log_mag_spect.shape)
        cepstrum_window[sample_cutoff:] = 0
        modified_cepstrum = cepstrum_window*cepstrum
        spec_env = np.real(ifft(modified_cepstrum))
        spec_env -= np.max(spec_env)
        return ModifiedMagSpectrum(spec_env, f)

    def to_Noise(self, sig_dur, fs):
        FFT_mag = np.abs(self.fft)
        resampled_mag = resample(FFT_mag, round(sig_dur*fs))
        random_phases = 2*pi*(np.random.rand(len(resampled_mag)) - 0.5)
        new_FFT = resampled_mag*np.exp(1j*random_phases)
        new_data = np.real(ifft(new_FFT))
        return Sound(new_data, fs)

    def display(self, ax, fscale="log"):
        log_mag_spectrum = self.log_mag_spect
        f = self.f

        if fscale == "log":
            ax.semilogx(f, log_mag_spectrum, alpha=0.5)
        elif fscale == "linear":
            ax.plot(f, log_mag_spectrum)
        else:
            raise ValueError("fscale should be 'log' or 'linear'")
        ax.set_title("Magnitude spectrum")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Relative magnitude [dB]")
        ax.set_xlim((f[0], f[-1]))
        ax.grid(linestyle=":")


class ModifiedMagSpectrum(MagnitudeSpectrum):
    def __init__(self, log_mag_spect, f):
        self.log_mag_spect = log_mag_spect
        self.f = f


class TFRepresentation(Representation):
    def display(self):
        pass


class STFT(TFRepresentation):
    def __init__(self, sound, win_dur, win_overlap=0.5, win_func=hanning):
        sig_len = len(sound)
        data = sound.data
        fs = sound.fs
        win_len = floor(fs*win_dur)
        adv_len = floor(win_len*(1 - win_overlap))
        n_wins = floor((len(sound) - win_len)/adv_len)
        window = win_func(win_len)

        spec_len = win_len//2 + 1
        stft = np.zeros((spec_len, n_wins), dtype=np.complex)
        for win_idx in range(n_wins):
            beg_idx = win_idx*adv_len
            end_idx = beg_idx + win_len

            curr_data = window*data[beg_idx:end_idx]
            stft[:, win_idx] = rfft(curr_data)

        self.t = np.linspace(0, sig_len/fs, n_wins)
        self.f = fs*rfftfreq(win_len)
        self.stft = stft
        self.mag_stft = np.abs(self.stft)
        log_mag_stft = 10*np.log10(self.mag_stft)
        log_mag_stft[log_mag_stft == -np.inf] = np.min(log_mag_stft[log_mag_stft != -np.inf])
        self.log_mag_stft = log_mag_stft
        self.fs = fs
        self.sig_len = sig_len
        self.win_len = win_len
        self.adv_len = adv_len
        self.n_wins = n_wins
        self.window = window

    def __gt__(self, other):
        if isinstance(other, STFT):
            mask = np.greater(self.log_mag_stft, other.log_mag_stft)
        else:
            raise RuntimeError("ill-defined comparison between spectrogram and non-spectrogram objects")
        return IdealBinaryMask(mask, self.t, self.f)

    def __lt__(self, other):
        return other.__gt__(self)

    def __add__(self, other):
        if isinstance(other, STFT):
            modified_log_mag_stft = self.log_mag_stft + other.log_mag_stft
        elif isinstance(other, int) or isinstance(other, float):
            modified_log_mag_stft = self.log_mag_stft + other
        else:
            raise RuntimeError("addition is only defined for spectrograms and real numbers")
        return ModifiedSTFT(
                self.stft, self.mag_stft, modified_log_mag_stft,
                self.fs, self.t, self.f,
                self.sig_len, self.win_len, self.adv_len,
                self.n_wins, self.window)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, IdealBinaryMask):
            modified_stft = other.mask*self.stft
            modified_mag_stft = other.mask*self.mag_stft
            modified_log_mag_stft = other.mask*self.log_mag_stft
        else:
            raise RuntimeError("multiplication for spectrograms is only defined for ideal binary masks")
        return ModifiedSTFT(
                modified_stft, modified_mag_stft, modified_log_mag_stft,
                self.fs, self.t, self.f,
                self.sig_len, self.win_len, self.adv_len,
                self.n_wins, self.window)

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_Sound(self, method="istft", delta_rmse_threshold=0.001, max_iter=1000, win_func=hanning):
        mag_stft = self.mag_stft
        stft = self.stft
        fs = self.fs
        sig_len = self.sig_len
        win_len = self.win_len
        adv_len = self.adv_len
        n_wins = self.n_wins

        if method == "istft":
            reconstructed_signal = np.zeros(sig_len)
            for win_idx in range(n_wins):
                beg_idx = win_idx*adv_len
                end_idx = beg_idx + win_len

                curr_spec = stft[:, win_idx]
                reconstructed_signal[beg_idx:end_idx] += np.real(irfft(curr_spec))
        elif method == "GLA":
            prev_rmse = np.finfo(np.float32).max
            curr_delta_rmse = 1.
            curr_iter = 0
            window = win_func(win_len)
            prev_reconstruction  = np.random.randn(sig_len)
            reconstructed_signal = np.random.randn(sig_len)
            while curr_iter < max_iter and curr_delta_rmse > delta_rmse_threshold:
                curr_phases = np.zeros((win_len//2 + 1, n_wins))
                for win_idx in range(n_wins):
                    beg_idx = win_idx*adv_len
                    end_idx = beg_idx + win_len

                    curr_sig = window*reconstructed_signal[beg_idx:end_idx]
                    curr_phases[:, win_idx] = np.angle(np.fft.rfft(curr_sig))
                proposal_stft = mag_stft*np.exp(1j*curr_phases)

                curr_sig = np.zeros(sig_len)
                for win_idx in range(n_wins):
                    beg_idx = win_idx*adv_len
                    end_idx = beg_idx + win_len

                    curr_spec = proposal_stft[:, win_idx]
                    curr_sig[beg_idx:end_idx] += window*np.real(irfft(curr_spec, win_len))
                reconstructed_signal = curr_sig

                curr_rmse = RMS(reconstructed_signal - prev_reconstruction)
                curr_delta_rmse = np.abs(prev_rmse - curr_rmse)/prev_rmse

                prev_reconstruction  = reconstructed_signal
                prev_rmse = curr_rmse
                curr_iter += 1
        else:
            raise ValueError("method must be 'istft' or 'GLA'")
        rms_val = RMS(reconstructed_signal)
        return Sound(reconstructed_signal/rms_val, fs)


    def display(self, ax, n_xticks=6, n_yticks=6):
        log_mag_stft = self.log_mag_stft
        t = self.t
        f = self.f

        ax.imshow(log_mag_stft, origin="lower", aspect="auto", cmap=DEFAULT_CMAP)
        ax.set_title("Spectrogram")

        xtick_locs = np.linspace(-0.5, t.size - 0.5, n_xticks, endpoint=True)
        xtick_vals = np.linspace(t[0], t[-1], n_xticks, endpoint=True)
        xtick_strs = ["{:.1f}".format(t_i) for t_i in xtick_vals]
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_strs)
        ax.set_xlabel("Time [s]")

        ytick_locs = np.linspace(-0.5, f.size - 0.5, n_yticks, endpoint=True)
        ytick_vals = np.linspace(f[0]/1000, f[-1]/1000, n_yticks, endpoint=True)
        ytick_strs = ["{:.1f}".format(t_i) for t_i in ytick_vals]
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_strs)
        ax.set_ylabel("Frequency [kHz]")



class IdealBinaryMask(TFRepresentation):
    def __init__(self, mask, t, f):
        self.t = t
        self.f = f
        self.mask = mask

    def display(self, ax, n_xticks=6, n_yticks=6):
        mask = self.mask
        t = self.t
        f = self.f

        ax.imshow(mask, origin="lower", aspect="auto", cmap=cm.Greys_r)
        ax.set_title("Ideal binary mask")

        xtick_locs = np.linspace(-0.5, t.size - 0.5, n_xticks, endpoint=True)
        xtick_vals = np.linspace(t[0], t[-1], n_xticks, endpoint=True)
        xtick_strs = ["{:.1f}".format(t_i) for t_i in xtick_vals]
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_strs)
        ax.set_xlabel("Time [s]")

        ytick_locs = np.linspace(-0.5, f.size - 0.5, n_yticks, endpoint=True)
        ytick_vals = np.linspace(f[0]/1000, f[-1]/1000, n_yticks, endpoint=True)
        ytick_strs = ["{:.1f}".format(t_i) for t_i in ytick_vals]
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_strs)
        ax.set_ylabel("Frequency [kHz]")



class ModifiedSTFT(STFT):
    def __init__(self, stft, mag_stft, log_mag_stft, fs, t, f,
                 sig_len, win_len, adv_len, n_wins, window):
        self.stft = stft
        self.mag_stft = mag_stft
        self.log_mag_stft = log_mag_stft
        self.fs = fs
        self.t = t
        self.f = f
        self.sig_len = sig_len
        self.win_len = win_len
        self.adv_len = adv_len
        self.n_wins = n_wins
        self.window = window



class ModulationSpectrum(Representation):
    def __init__(self, STFT_obj):
        log_mag_stft = STFT_obj.log_mag_stft
        f_n, t_n = log_mag_stft.shape

        mag_spect2D = np.abs(fft2(log_mag_stft))
        log_mag_spect2D = 10*np.log10(mag_spect2D)
        mod_spect = np.roll(log_mag_spect2D[:f_n//2, :], (0, t_n//2))
        self.mod_spect = mod_spect

        t_max = STFT_obj.t[-1]
        w_t_n = mod_spect.shape[1]
        w_t_max = w_t_n/t_max
        w_t = np.linspace(0, w_t_max, w_t_n)
        self.w_t = w_t - np.mean(w_t)

        fs = STFT_obj.fs
        w_f_n = mod_spect.shape[0]
        w_f_max = w_f_n/(fs/2/1000)
        self.w_f = np.linspace(0, w_f_max, w_f_n)


    def display(self, ax, n_xticks=7, n_yticks=6):
        mod_spect = self.mod_spect
        w_t = self.w_t
        w_f = self.w_f

        ax.imshow(mod_spect, origin="lower", aspect="auto", cmap=DEFAULT_CMAP)
        ax.set_title("Modulation spectrum")

        xtick_locs = np.linspace(-0.5, w_t.size - 0.5, n_xticks, endpoint=True)
        xtick_vals = np.linspace(w_t[0], w_t[-1], n_xticks, endpoint=True)
        xtick_strs = ["{:.1f}".format(t_i) for t_i in xtick_vals]
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_strs)
        ax.set_xlabel("Temporal modulation [Hz]")

        ytick_locs = np.linspace(-0.5, w_f.size - 0.5, n_yticks, endpoint=True)
        ytick_vals = np.linspace(w_f[0], w_f[-1], n_yticks, endpoint=True)
        ytick_strs = ["{:.1f}".format(t_i) for t_i in ytick_vals]
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_strs)
        ax.set_ylabel("Spectral modulation [cyc/kHz]")


    def to_ModifiedSTFT(self):
        raise NotImplementedError



class InterauralCues(Representation):
    # TODO: Implement interaural correlogram
    def __init__(self, binaural_sound, win_dur, win_overlap=0.5,
                 suppression_threshold_fraction=0.01):
        self.sound = binaural_sound
        data = binaural_sound.data
        fs = binaural_sound.fs

        sig_len = len(binaural_sound)
        win_len = floor(fs*win_dur)
        adv_len = floor(win_len*(1 - win_overlap))
        n_wins = floor((sig_len - win_len)/adv_len)

        avg_power_in_window = win_len/sig_len*RMS(data)**2
        suppression_threshold_power = suppression_threshold_fraction*avg_power_in_window

        tau = (np.arange(win_len) - win_len//2)/fs
        t = np.linspace(0, (sig_len - adv_len)/fs, n_wins)
        ITDs = np.zeros(n_wins)
        ILDs = np.zeros(n_wins)
        IACs = np.zeros(n_wins)
        for win_idx in range(n_wins):
            beg_idx = win_idx*adv_len
            end_idx = beg_idx + win_len

            curr_data = data[beg_idx:end_idx]
            curr_CCF = correlate(curr_data[:, 0], curr_data[:, 1], mode="same")/curr_data.shape[0]
            curr_avg_power = RMS(curr_data)**2
            if curr_avg_power < suppression_threshold_power:
                ITDs[win_idx] = 0.
                ILDs[win_idx] = 0.
                IACs[win_idx] = 0.
            else:
                ITDs[win_idx] = tau[np.argmax(curr_CCF)]
                curr_P_in_both = np.mean(np.square(curr_data), axis=0)
                curr_P_L = curr_P_in_both[0]
                curr_P_R = curr_P_in_both[1]
                ILDs[win_idx] = 10*np.log10(curr_P_R/curr_P_L)
                IACs[win_idx] = 0.

        self.t = t
        self.ITDs = ITDs
        self.ILDs = ILDs
        self.IACs = IACs


    def display(self, ax):
        t = self.t
        ITDs = self.ITDs
        ILDs = self.ILDs
        max_ITD = 1e6*max(np.abs(ITDs))
        max_ILD = max(np.abs(ILDs))

        ax.plot(t, 1e6*ITDs, color="m", alpha=0.5, label="ITD")
        ax.hlines(0, t[0], t[-1], color="k", linestyle="-", alpha=0.25)
        ax.set_title("Interaural cues")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"ITD [$\mu$s]")
        ax.set_xlim((t[0], t[-1]))
        ax.set_ylim((-1.1*max_ITD, 1.1*max_ITD))
        ax.grid(linestyle=":")
        ax.legend(loc=2)

        twin_ax = ax.twinx()
        twin_ax.plot(t, ILDs, color="g", alpha=0.5, label="ILD")
        twin_ax.set_ylabel("ILD [dB]")
        twin_ax.set_ylim((-1.1*max_ILD, 1.1*max_ILD))
        twin_ax.legend(loc=1)


def display_STFT(x, S, figsize=(12, 8)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    Waveform(x).display(axes[0, 0])
    MagnitudeSpectrum(x).display(axes[0, 1])
    S.display(axes[1, 0])
    ModulationSpectrum(S).display(axes[1, 1])
    fig.tight_layout()
    AudioControl(x).display()
