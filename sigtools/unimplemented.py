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
