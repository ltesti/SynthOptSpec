#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import types

def cardelli_extinction(wave, Av, Rv):
    # If you use it to apply a reddening to a spectrum, multiply it for the result of
    # this function, while you should divide by it in the case you want to deredden it.

    #ebv = Av/Rv

    x = 10000./ wave # Convert to inverse microns
    npts = len(x)
    a = np.zeros(npts)
    b = np.zeros(npts)
    #******************************

    good = (x > 0.3) & (x < 1.1) #Infrared
    Ngood = np.count_nonzero(good == True)
    if Ngood > 0:
        a[good] = 0.574 * x[good]**(1.61)
        b[good] = -0.527 * x[good]**(1.61)

    #******************************
    good = (x >= 1.1) & (x < 3.3) #Optical/NIR
    Ngood = np.count_nonzero(good == True)
    if Ngood > 0: #Use new constants from O'Donnell (1994)
        y = x[good] - 1.82
        c1 = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0] #New coefficients
        c2 = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0] #from O'Donnell (1994)

        a[good] = np.polyval(c1,y)
        b[good] = np.polyval(c2,y)

    A_lambda = Av * (a + b/Rv)

    ratio = 10.**(-0.4*A_lambda)

    return ratio