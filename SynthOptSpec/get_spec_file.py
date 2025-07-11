#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import types

from .extinction import cardelli_extinction
from .resample import resamp_spec

def nrefrac(wl, density=1.0):
   """Calculate refractive index of air from Cauchy formula.

   Input: wavelength in Angstrom, density of air in amagat (relative to STP,
   e.g. ~10% decrease per 1000m above sea level).
   Returns N = (n-1) * 1.e6.

   directly taken from: https://phoenix.ens-lyon.fr/Grids/FORMAT
   
   Note that Phoenix delivers synthetic spectra in the vaccum and that a line
   shift is necessary to adapt these synthetic spectra for comparisons to
   observations from the ground. For this, divide the vacuum wavelengths by
   (1+1.e-6*nrefrac) as returned from the function below to get the air 
   wavelengths (or use the equation for AIR from it).  

   """

   # The IAU standard for conversion from air to vacuum wavelengths is given
   # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
   # Angstroms, convert to air wavelength (AIR) via:

   #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

   #try:
   #    if isinstance(wavelength, types.ObjectType):
   #        wl = np.array(wavelength)
   #except TypeError:
   #    return None

   wl2inv = (1.e4/wl)**2
   refracstp = 272.643 + 1.2288 * wl2inv  + 3.555e-2 * wl2inv**2
   return density * refracstp

def get_spec_file(teff,LogG,modspecdir=None,oldgrid=False,model='Settl', in_dict=None):
    """
    This function takes effective temperature (teff) and Log(g) (LogG) for the synthetic spectra
    and returns the filename for the spectrum. The directory where the synthetic spectra reside 
    is also a parameter.
    """
    teffstr = str(int(teff/100))
    Z = '0.0'
    modflux_log = True

    if in_dict:
        modspecdir = in_dict['modspecdir']
        modflux_log = False
        if teff<1000.:
            teffstr = '00'+teffstr
        elif teff<10000.:
            teffstr = '0'+teffstr
        if teff>00.:
            specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-'+in_dict['model']+'.spec.fits'

    #modflux_log=False
    if oldgrid:
        if not modspecdir:
            modspecdir = 'Models/bt-settl-fits/'
        teffstr = '0' + teffstr
        specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.fits'
    else:
        #modspecdir = 'Models/bt-settl/'
        if teff<1000.:
            teffstr = '00'+teffstr
        elif teff<10000.:
            teffstr = '0'+teffstr
        if teff>00.:
            if model == 'Dusty':
                if not modspecdir:
                    modspecdir = 'Models/bt-dusty/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + '.BT-Dusty.spec.7'
            elif model == 'Dusty-last':
                    if not modspecdir:
                        modspecdir = 'Models/bt-dusty-last/'
                    specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + '.BT-Dusty.spec.7'
            elif model == 'Dusty-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-dusty-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Dusty.7.dat.txt'
            elif model == 'Dusty-bds':
                if not modspecdir:
                    modspecdir = 'Models/bt-dusty-bds/'
                modflux_log = False
                if teff >= 2600.:
                    specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Dusty.7.dat.txt'
                else:
                    specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + '.BT-Dusty.7.dat.txt'
            elif model == 'Settl':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl/'
                specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.edit'
            elif model == 'Settl-last':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-last/'
                specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7'
            elif model == 'Settl-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.dat.txt'
            elif model == 'Settl-ffpmos': #added for VBianchet thesis
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-ffpmos/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-0.0.BT-Settl.spec.7.dat.txt'
            elif model == 'Settl-2019':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-2019/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.edit'
            elif model == 'NextGen':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7'
            elif model == 'NextGen-last':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen-last/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7'
            elif model == 'NextGen-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7.dat.txt'
            elif model == 'NextGen-agss2009':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen-agss2009/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7.dat.txt'
            #elif model == 'Kurucz':
            #    if not modspecdir:
            #        modspecdir = 'Models/Kurucz2003all/'
            #    modflux_log = False
            #    specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7.dat.txt'
            elif model == 'Cond':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Cond.7'
            elif model == 'Cond-last':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond-last/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Cond.7'
            elif model == 'Cond-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Cond.7.dat.txt'
            elif model == 'Cond-bds':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond-bds/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + '.BT-Cond.7.dat.txt'
            elif model == 'Settl-fits':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-fits/'
                specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.fits'
            else:
                if not modspecdir:
                    modspecdir = 'Models/bt-settl/'
                specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.edit'
    #

    return specfile, modflux_log

def get_spec_file_old(teff,LogG,modspecdir='Models/bt-settl/'):
    """
    This function takes effective temperature (teff) and Log(g) (LogG) for the synthetic spectra
    and returns the filename for the spectrum. The directory where the synthetic spectra reside
    is also a parameter.
    """
    teffstr = str(int(teff/100))
    modflux_log = True
    if teff<1000.:
        teffstr = '00'+teffstr
    elif teff<10000.:
        teffstr = '0'+teffstr
    Z = '0.0'
    if teff>2500.:
        specfile = modspecdir+'lte'+teffstr+'-'+LogG+'-'+Z+'a+0.0.BT-NextGen.7.dat.txt'
    else:
        specfile = modspecdir+'lte'+teffstr+'-'+LogG+'-'+Z+'.BT-Settl.7.dat.txt'
    return specfile, modflux_log
