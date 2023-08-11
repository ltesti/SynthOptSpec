#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import (division, print_function, absolute_import,
#                        unicode_literals)

import numpy as np
# import types

def ComputeMag(wl, spec, wlf, traspf, fzero):
    """Compute magnitude for filter transparency
    
    The function computed the magnitude corresponding to the transparency curve
    from the spectrum. It is important to ensure that the units are consistent 
    among the various quantities (wavelength, fluxes and and zero point flux)
    
    :params wl: (np.array float) wavelength for the input spectrum
    :params spec: (np.array float) input spectrum values
    :params wlf: (np.array float) wavelength grid for the transparency
    :params traspf: (np.array float) transparency function
    :params fzero: (float) zero flux for the magnitude
    :return: (float) computed magnitude
    """
    
    # First choice to be made:
    #   use the filter as wavelength base or the spectrum?
    #   we chose to use the finer grid
    #   and restrict to the range of the filter grid
    #
    #   TODO:
    #       need to throw an exception if spec is not
    #       wider than the filter
    
    nfs = np.argsort(wlf)
    
    nfilt = np.where((wl >= wlf[nfs[0]]) & (wl <= wlf[nfs[-1]]))
    nfss = np.argsort(wl[nfilt])
    
    if (wl[1]-wl[0])**2 < (wlf[1]-wlf[0])**2: #spectrum is on a finer grid than filter
        wli = np.copy((wl[nfilt])[nfss])
        fli = np.copy((spec[nfilt])[nfss])
        wl2 = np.copy(wlf[nfs])
        fl2 = np.copy(traspf[nfs])
    else:
        wli = np.copy(wlf[nfs])
        fli = np.copy(traspf[nfs])
        wl2 = np.copy((wl[nfilt])[nfss])
        fl2 = np.copy((spec[nfilt])[nfss])
        
    # interpolate the less finer grid into the finer grid
    #   
    fl2i = np.interp(wli,wl2,fl2)
    
    fint = np.trapz(fli*fl2i,wli)/np.trapz(traspf[nfs],wlf[nfs])
    
    return -2.5*np.log10(fint/fzero)
    