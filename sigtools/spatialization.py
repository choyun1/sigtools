#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""spatialization

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""

from tqdm import tqdm

from math import ceil, floor
import numpy as np
from numpy import pi, sin, cos, tan, arctan, arctan2
from numpy.fft import irfft
from scipy.spatial import Delaunay
from scipy.signal import resample
from scipy.signal import convolve as sp_convolve

from sigtools.utils import *
from sigtools.sounds import *


def rect_to_sph(x, y, z):
    hxy = np.hypot(x, y)
    rho = np.hypot(hxy, z)
    theta = arctan2(z, hxy)
    phi = arctan2(y, x)
    return rho, theta, phi


def sph_to_rect(rho, theta, phi):
    x = rho*cos(theta)*cos(phi)
    y = rho*cos(theta)*sin(phi)
    z = rho*sin(theta)
    return x, y, z


def rect_to_hcc(x, y, z):
    rho, theta, phi = rect_to_sph(x, y, z)
    dist = 100*rho
    elev = 180*theta/pi
    azim = np.mod(180*phi/pi - 90, 360)
    return dist, elev, azim


def hcc_to_rect(dist, elev, azim):
    rho = dist/100
    theta = pi*elev/180
    phi = pi*np.mod(90 - azim, 360)/180
    return sph_to_rect(rho, theta, phi)


def sph_to_hcc(rho, theta, phi):
    return rect_to_hcc(*sph_to_rect(rho, theta, phi))


def hcc_to_sph(dist, elev, azim):
    return rect_to_sph(*hcc_to_rect(dist, elev, azim))


def tuple_to_array(seq_of_coord_tuples):
    return np.array(list(zip(*seq_of_coord_tuples))).T


def make_linear_trajectory(beg_pt, end_pt, n_spatial_samples):
    ndim = len(beg_pt)
    return np.array([np.linspace(beg_pt[dim_i],
                                 end_pt[dim_i],
                                 n_spatial_samples)
                     for dim_i in range(ndim)]).T


def make_hcc_circular_trajectory(beg_pt, end_pt, n_spatial_samples):
    ndim = len(beg_pt)
    hcc_coords = np.array([np.linspace(beg_pt[dim_i],
                                       end_pt[dim_i],
                                       n_spatial_samples)
                           for dim_i in range(ndim)]).T
    hcc_coords_tuple = list(map(tuple, hcc_coords))
    rect_coords_tuple = [hcc_to_rect(*coords) for coords in hcc_coords_tuple]
    rect_coords = tuple_to_array(rect_coords_tuple)
    return rect_coords


class PKU_IOA_HRIR:
    def __init__(self):
        dir = "/home/acho/Sync/Sounds/HRTF-PKU-IOA/all_HRIRs/"
        fs = 65536
        paths  = []
        hcc_coords = []
        dists = [20, 30, 40, 50, 75, 100, 130, 160]
        elevs = range(-40, 91, 10)
        for dist in dists:
            for elev in elevs:
                if elev == 60:
                    azims = range(0, 361, 10)
                elif elev == 70:
                    azims = range(0, 361, 15)
                elif elev == 80:
                    azims = range(0, 361, 30)
                elif elev == 90:
                    azims = range(0, 361, 360)
                else:
                    azims = range(0, 361, 5)
                for azim in azims:
                    curr_name =   "azi" + str(azim)\
                              + "_elev" + str(elev)\
                              + "_dist" + str(dist)\
                              + ".dat"
                    paths.append(dir + curr_name)
                    hcc_coords.append((dist, elev, azim))
        rect_coords = tuple_to_array([hcc_to_rect(*pt) for pt in hcc_coords])
        self.tessel = Delaunay(rect_coords)
        self.fs = fs
        self.paths = paths

    def compute_HRIRs(self, rect_coords, new_fs):
        tessel = self.tessel
        paths = self.paths
        old_fs = self.fs

        # Compute barycentric coordinates
        N, ndim = rect_coords.shape
        tessel_idxs = tessel.find_simplex(rect_coords).astype(int)
        if np.any(tessel_idxs < 0):
            raise ValueError("invalid coordinates; cannot interpolate")
        b_coords = np.zeros((N, ndim + 1))
        T_inv = tessel.transform[tessel_idxs, :ndim, :ndim]
        r     = tessel.transform[tessel_idxs, ndim, :]
        for i in range(N):
            partial_b = T_inv[i, :, :].dot( (rect_coords - r).T[:, i] )
            b_coords[i, :] = np.append(partial_b, 1 - np.sum(partial_b))

        # Interpolate using the barycentric coordinates
        n_HRIRs, n_coords = b_coords.shape
        HRIR_idxs = tessel.simplices[tessel_idxs].astype(int)
        all_HRIRs = []
        for i in range(n_HRIRs):
            weighted_HRIRs = np.array( [b_coords[i, j]*np.fromfile(paths[HRIR_idxs[i, j]]).reshape(2, -1).T
                                        for j in range(n_coords)] )
            curr_HRIR = np.sum(weighted_HRIRs, axis=0)
            # Resample
            old_n_samples = curr_HRIR.shape[0]
            new_n_samples = floor(old_n_samples*new_fs/old_fs)
            resampled_HRIR = resample(curr_HRIR, new_n_samples)
            all_HRIRs.append(HRIR(resampled_HRIR, new_fs))
        return all_HRIRs


PKU_IOA_DATABASE = PKU_IOA_HRIR()


def move_sound(trajectory, sound):
    n_spatial_samples = trajectory.shape[0]
    data = sound.data
    fs = sound.fs
    all_HRIRs = PKU_IOA_DATABASE.compute_HRIRs(trajectory, fs)

    sig_len = len(sound)
    block_size = floor(sig_len/n_spatial_samples)
    moving_sound_data = np.zeros( (sig_len + len(all_HRIRs[0]) - 1, 2) )
    # for i in tqdm(range(n_spatial_samples)):
    for i in range(n_spatial_samples):
        curr_beg_idx = i*block_size
        curr_end_idx = (i + 1)*block_size

        curr_HRIR_data = all_HRIRs[i].data
        curr_block = np.zeros((sig_len, 2))
        curr_block[curr_beg_idx:curr_end_idx] = data[curr_beg_idx:curr_end_idx]
        moving_sound_data += np.array([sp_convolve(curr_HRIR_data[:, j],
                                                   curr_block[:, j])
                                       for j in range(2)]).T
    rms_val = RMS(moving_sound_data)
    return Sound(moving_sound_data/rms_val, fs)
