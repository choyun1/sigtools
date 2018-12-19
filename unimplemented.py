#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""sound

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""


# def make_modulation_filterbank(f_c, f_lo, f_hi):
#     """Filter coefficients from Dau et al., 1997"""
#     # TODO
#     from scipy.signal import freqz
#     f_res = 20
#     Q = 2
#     B = f_res/Q
#     fs = 400
#     Delta = 1/fs
#
#     b = np.array([1 - np.exp(-np.pi*B*Delta)])
#     a = np.array([1, np.exp(-np.pi*B*Delta)*np.exp(-1j*2*np.pi*f_res*Delta)])
#     w, h = freqz(b, a)
#     return 0
#
#
# def subband_resampled_envelopes(subbands, fs_orig, fs_resample):
#     from scipy.signal import hilbert, decimate
#
#     q = int(fs_orig/fs_resample)
#     envelope = np.abs(hilbert(subbands, axis=0))
#     tfs = np.divide(subbands, envelope, axis=1)
#     compressed_env = np.power(envelope, 0.3)
#     resampled_env  = decimate(compressed_env, q, axis=0)
#     return tfs, resampled_env
#
#
# def subband_envs_marginal_moments(subband_envs):
#     weight_function = 1/subband_envs.shape[0]
#     M1 = np.sum(weight_function*subband_envs, axis=0)
#     M2 = np.sum(weight_function*(subband_envs - M1)**2, axis=0)/M1**2
#     sigma = np.sqrt(M2*M1**2)
#     M3 = np.sum(weight_function*(subband_envs - M1)**3, axis=0)/sigma**3
#     M4 = np.sum(weight_function*(subband_envs - M1)**4, axis=0)/sigma**4
#     return np.concatenate( (M1, M2, M3, M4) )
#
#
# def subband_cross_correlations(subband_envs, channel_dists_to_consider=(1, 2, 3, 5, 8, 11, 16, 21)):
#     weight_function = 1/subband_envs.shape[0]
#     channel_idxs = range(subband_envs.shape[1])
#     channel_pairs = [(j, k) for j in channel_idxs for k in channel_idxs if (k - j) in channel_dists_to_consider]
#
#     def pairwise_channel_correlation(channel_j, channel_k):
#         return np.sum( weight_function*(channel_j - np.mean(channel_j))*(channel_k - np.mean(channel_k))/\
#                                        (np.std(channel_j)*np.std(channel_k)) )
#     cross_corrs = np.array( [pairwise_channel_correlation(subband_envs[:, j], subband_envs[:, k]) for j, k in channel_pairs] )
#     return channel_pairs, cross_corrs
#
#
# # def subband_modulation_power(single_subband_env, subband_env_var, mod_filterbank):
# #     from scipy.fftpack import next_fast_len, rfft, irfft
# #     sig_len = single_subband_env.shape[0]
# #     filt_len, n_channels = mod_filterbank.shape[1]
# #     if sig_len != filt_len:
# #         raise ValueError("signal length must equal filter length")
#
# #     weight_function = 1/sig_len
# #     fft_len = next_fast_len(sig_len)
# #     fft_sig = rfft(single_subband_env)[:sig_len].reshape(sig_len, 1)
# #     filtered_subbands = filterbank*np.matlib.repmat(fft_sig, 1, n_channels)
# #     modulation_subbands = irfft(filtered_subbands, fft_len, axis=0)[:sig_len, :]
#
# #     return np.sum(weight*function*modulation_subbands**2, axis=0)/subband_env_var
#
# def subband_modulation_power(single_subband_env, subband_env_var, fs, mod_channels, mod_f_lo, mod_f_hi):
#     # TODO
#     sig_len = len(single_subband_env)
#     sigtools.make_overlapping_windows(
#                          sig_len=sig_len,
#                          scale="ERB",
#                          n_channels=mod_channels,
#                          fs=fs,
#                          lo_lim_freq=mod_f_lo,
#                          hi_lim_freq=mod_f_hi,
#                          win_shape=lambda x: np.cos(x/2))
#     return 0
#
#
# def subband_modulation_corrs(single_subband_env, mod_filterbank):
#     # TODO
#     return 0
#
#
# def calculate_summary_statistics(sound):
#     tfs, subband_envs = subband_resampled_envelopes(sound.subbands, fs, 400)
#     marginal_moments = subband_envs_marginal_moments(subband_envs)
#     channel_pairs, cross_corr = subband_cross_correlations(subband_envs)
#     return np.concatenate( (marginal_moments, cross_corr) )
#
#
# def summary_statistics_cost_function(reference_sound, curr_sound):
#     # TODO
#     reference_statistics = 0
#     return np.sum((reference_statistics - calculate_summary_statistics(curr_sound))**2)
