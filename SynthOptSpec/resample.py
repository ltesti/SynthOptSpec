#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import types

def resamp_spec(wlsamp, wl, fl):
    """
    Function to resample an input spectrum on a new wavelength grid
    the assumption is to do a simple binning - average flux per wl bin
    """
    rfl = np.zeros(len(wlsamp))

    bins = np.zeros((len(wlsamp), 2))
    for i in range(len(wlsamp)):
        # set up the bins
        if i < len(wlsamp) - 1:
            db = (wlsamp[i + 1] - wlsamp[i]) / 2.
            bins[i, 1] = wlsamp[i] + db
            bins[i + 1, 0] = wlsamp[i] + db
            if i == 0:
                bins[i, 0] = wlsamp[i] - db
            if i == len(wlsamp) - 2:
                bins[i + 1, 1] = wlsamp[i + 1] + db
        # find who is in i-bin 
        nib = np.where((wl >= bins[i, 0]) & (wl < bins[i, 1]))
        if len(nib[0] > 0):
            rfl[i] = np.mean(fl[nib])
    #
    return rfl

