#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""utils

--------------------------
@author: Adrian Y. Cho
@email:  aycho@g.harvard.edu
--------------------------
"""

import numpy as np


def RMS(data):
    return np.sqrt(np.mean(np.square(data)))


def freq_to_ERB(freq):
    return 9.265*np.log(1 + freq/(24.7*9.265))


def ERB_to_freq(ERB):
    return 24.7*9.265*(np.exp(ERB/9.265) - 1)
