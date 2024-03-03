# cython : language_level=3

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import scipy.interpolate as ssi
cimport numpy as cnp
cimport cython
cnp.import_array()

cpdef resamp_spec_c(cnp.ndarray wlsamp, cnp.ndarray wl, cnp.ndarray fl):
    """
    Function to resample an input spectrum on a new wavelength grid
    the assumption is to do a simple binning - average flux per wl bin
    """

    cdef int i
    cdef float db
    cdef cnp.ndarray rfl = np.zeros(len(wlsamp), dtype=np.float)

    cdef cnp.ndarray bins = np.zeros((len(wlsamp), 2), dtype=np.float)

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

cpdef fgauss_c(float x, float s):
    return np.exp(-x**2/(2.*s**2))

cpdef smooth_c(cnp.ndarray wl, cnp.ndarray fl, float R, bint muse_res_manual, bint verbose, float nsig):

    # These are from the MUSE manual Section 3.2 (P110 version)
    cdef cnp.ndarray muse_res_wl = np.array([4650.0,5000.0,5500.0,6000.0,6500.0,7000.0,7500.0,8000.0,8500.0,9000.0,9350.0], dtype=np.float)
    cdef cnp.ndarray muse_res = np.array([1609.,1750.,1978.,2227.,2484.,2737.,2975.,3183.,3350.,3465.,3506.], dtype=np.float)

    cdef cnp.ndarray ssfl = np.zeros(len(fl), dtype=np.float)
    cdef cnp.ndarray dl = wl / R
    cdef cnp.ndarray sdl = 0.0*dl

    cdef int i, j
    cdef float msdl, psdl, fs, fg, area

    if muse_res_manual == 0:
        if verbose == 0: print("Using MUSE-res-manual ({0}, {1}) => ".format(dl[0],dl[-1]))
        muse_res_interp = ssi.interp1d(muse_res_wl, muse_res)
        nlow = np.where(wl<=muse_res_wl[0])
        dl[nlow] = wl[nlow]/muse_res[0]
        nhigh = np.where(wl>=muse_res_wl[-1])
        dl[nhigh] = wl[nhigh]/muse_res[-1]
        ninterp = np.where((wl>muse_res_wl[0]) & (wl<muse_res_wl[-1]))
        dl[ninterp] = wl[ninterp]/muse_res_interp(wl[ninterp])
        if verbose == 0: print(" ({0}, {1})\n".format(dl[0], dl[-1]))

    sdl = dl / (2. * np.sqrt(2. * np.log(2.)))

    for i in range(len(fl)):
        #dl = self.aswl[i]/R
        #nsig = 5.
        #sdl = dl/(2.*np.sqrt(2.*np.log(2.)))
        msdl = wl[i]-nsig*sdl[i]
        psdl = wl[i]+nsig*sdl[i]
        nsm = np.where((wl >= msdl) & (wl<=psdl))
        smwl = wl[nsm]
        fs = 0.
        area = 0.
        for j in range(len(smwl)):
            fg = fgauss_c(smwl[j]-wl[i],sdl[i])
            fs += fg*fl[nsm[0][j]]
            area += fg
        ssfl[i] = fs/area
    #
    return ssfl
