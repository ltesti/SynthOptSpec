#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import types


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
        if teff<1000.:
            teffstr = '00'+teffstr
        elif teff<10000.:
            teffstr = '0'+teffstr
        if teff>00.:
            specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-'+in_dict['model']+'.spec.fits'

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
            elif model == 'Dusty-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-dusty-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Dusty.7.dat.txt'
            elif model == 'Settl':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl/'
                specfile = modspecdir + 'lte' + teffstr + '.0-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.edit'
            elif model == 'Settl-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.dat.txt'
            elif model == 'Settl-2019':
                if not modspecdir:
                    modspecdir = 'Models/bt-settl-2019/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Settl.spec.7.edit'
            elif model == 'NextGen':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7'
            elif model == 'NextGen-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-nextgen-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-NextGen.7.dat.txt'
            elif model == 'Cond':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond/'
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Cond.7'
            elif model == 'Cond-restricted':
                if not modspecdir:
                    modspecdir = 'Models/bt-cond-restricted/'
                modflux_log = False
                specfile = modspecdir + 'lte' + teffstr + '-' + LogG + '-' + Z + 'a+0.0.BT-Cond.7.dat.txt'
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

def get_g_t_idx(df,myg,myt):
    """
    Given a value of LogG (myg) and Teff (myt), this function contains the logic to identify 
    the closest LogG and Teff (two per LogG) in the available spectral library.
    """

    lib_logg = df['LogG'].unique()
    lib_teff = []
    for logg in lib_logg:
        lib_teff.append(df.iloc[np.where(np.array(df['LogG']) == logg)]['Teff'].unique())

    inlogg = False
    inteff = False
    exactg = False
    exacttg1 = False
    exacttg2 = False

    if (myg >= lib_logg[0]) & (myg <= lib_logg[-1]):
        inlogg = True
        myg_idx = np.where(lib_logg == myg)
        idxg = np.searchsorted(lib_logg,myg,'left') - 1
        if len(myg_idx[0]) == 0:
            exactg = False
            idxg = np.searchsorted(lib_logg,myg,'left') -1
        else:
            exactg = True
            idxg = myg_idx[0][0]
        if (myt >= lib_teff[idxg][0]) & (myt <= lib_teff[idxg][-1]):
            inteff = True
            myt_idx = np.where(lib_teff[idxg] == myt) 
            if len(myt_idx[0]) == 0:
                exacttg1 = False
                idxtg1 = np.searchsorted(lib_teff[idxg],myt,'left')-1
            else:
                exacttg1 = True
                idxtg1 = myt_idx[0][0]
            if not exactg:
                if (myt < lib_teff[idxg+1][0]) & (myt > lib_teff[idxg+1][-1]):
                    inteff = False
                else:
                    myt_idx = np.where(lib_teff[idxg+1] == myt) 
                    if len(myt_idx[0]) == 0:
                        exacttg2 = False
                        idxtg2 = np.searchsorted(lib_teff[idxg+1],myt,'left')-1
                    else:
                        exacttg2 = True
                        idxtg2 = myt_idx[0][0]
    #print("inlogg={0}, inteff={1}".format(inlogg,inteff))
    if inlogg and inteff:
        #print("exactg={0}, idxg={1}".format(exactg,idxg))
        #print("exacttg1={0}, idxtg1={1}".format(exacttg1,idxtg1))
        wltg1_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1]['wl']
        ftg1_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1]['f']
        tg1_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1]['Teff']
        g1 = lib_logg[idxg]
        if not exacttg1:
            wltg1_2  = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1+1]['wl']
            ftg1_2 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1+1]['f']
            tg1_2 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg])]).iloc[idxtg1+1]['Teff']
        else:
            wltg1_2 = None
            ftg1_2 = None
            tg1_2 = None
        if not exactg:
            #print("exacttg2={0}, idxtg2={1}".format(exacttg2,idxtg2))
            g2 = lib_logg[idxg+1]
            wltg2_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2]['wl']
            ftg2_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2]['f']
            tg2_1 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2]['Teff']
            if not exacttg2:
                wltg2_2  = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2+1]['wl']
                ftg2_2 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2+1]['f']
                tg2_2 = (df.iloc[np.where(np.array(df['LogG']) == lib_logg[idxg+1])]).iloc[idxtg2+1]['Teff']
            else:
                wltg2_2 = None
                ftg2_2 = None
                tg2_2 = None
        else:
            g2 = None
            wltg2_1 = None
            ftg2_1 = None
            tg2_1 = None
            wltg2_2 = None
            ftg2_2 = None
            tg2_2 = None
    else:
        exactg = False
        exacttg1 = False
        exacttg2 = False
        g1 = None
        wltg1_1 = None
        ftg1_1 = None
        tg1_1 = None
        wltg1_2 = None
        ftg1_2 = None
        tg1_2 = None
        g2 = None
        wltg2_1 = None
        ftg2_1 = None
        tg2_1 = None
        wltg2_2 = None
        ftg2_2 = None
        tg2_2 = None
        #print("Error: requested (LogG, Teff) outside bounds!")
    mydatadic = {
        'LogG' : myg,
        'Teff' : myt,
        'inlogg' : inlogg,
        'inteff' : inteff,
        'exactg' : exactg,
        'exacttg1' : exacttg1,
        'exacttg2' : exacttg2,
        'g1' : g1,
        'wltg1_1' : wltg1_1,
        'ftg1_1' : ftg1_1,
        'tg1_1' : tg1_1,
        'wltg1_2' : wltg1_2,
        'ftg1_2' : ftg1_2,
        'tg1_2' : tg1_2,
        'g2' : g2,
        'wltg2_1' : wltg2_1,
        'ftg2_1' : ftg2_1,
        'tg2_1' : tg2_1,
        'wltg2_2' : wltg2_2,
        'ftg2_2' : ftg2_2,
        'tg2_2' : tg2_2,
    }
    return mydatadic

def get_phot_spec(df, logg, teff):
    # gets the spectra from the library and interpolates at the resolution
    # of the lower resolution spectrum in the (up to) four spectra
    #
    data_for_interpolation = get_g_t_idx(df, logg, teff)
    wl, fl = get_interp_spec(data_for_interpolation)
    #
    return wl, fl

def apply_rvel(wl, rvel):
    #
    ckms = 299792.458  # speed of light in km/s
    # Radial velocity
    wl = wl * (1. + rvel / ckms)
    #
    return wl

def get_spec(df, logg, teff, wlmin=4750.1572265625, wlmax=9351.4072265625, dl=1.25, av=None, rv=3.1, rvel=None, normalization="Dominika"):
    #
    # gets the spectra from the library and interpolates at the resolution
    # of the lower resolution spectrum in the (up to) four spectra
    #
    wl, fl = get_phot_spec(df, logg, teff)

    # Radial velocity
    if rvel and rvel != 0.0:
         wl = apply_rvel(wl, rvel)

    # Extinction
    if av and av > 0.0:
        fl = fl*cardelli_extinction(wl, av, rv)

    # Resample to the desired spectral resolution
    if (dl*dl<(wl[1]-wl[0])**2.) & (dl*dl<(wl[-1]-wl[-2])**2.):
        print("Error: te spectral library does not have the required resolution")
        w = wl
        f = fl
    elif (wlmin < wl[0]) or (wlmax > wl[-1]):
        print("Error: te spectral library does not cover fully the requested wavelength range")
        w = wl
        f = fl
    else:
        w = np.arange(wlmin, wlmax, dl)
        f = resamp_spec(w, wl, fl)

    # Normalize and return the final spectrum
    if normalization:
        if normalization == "Dominika":
            id750 = np.abs(w - 7500.).argmin()
            f750 = np.nanmedian(f[id750 - 3:id750 + 3])
            fn = f / f750
    else:
        fn = np.copy(f)
    #
    return w, fn


def get_wlgrid(w1, w2):
    #
    wstart = max(w1[0], w2[0])
    wend = min(w1[-1], w2[-1])
    dw1 = max(np.sqrt((w1[1] - w1[0]) ** 2), np.sqrt((w1[-1] - w1[-2]) ** 2))
    dw2 = max(np.sqrt((w2[1] - w2[0]) ** 2), np.sqrt((w2[-1] - w2[-2]) ** 2))
    dw = max(dw1, dw2)
    return np.arange(wstart, wend, dw)


def get_tinterp(wltg, teff, w1, w2, f1, f2, t1, t2, logint=False):
    #
    rf1 = resamp_spec(wltg, w1, f1)
    rf2 = resamp_spec(wltg, w2, f2)
    if logint:
        k1 = 1. - (np.log10(teff) - np.log10(t1)) / (np.log10(t2) - np.log10(t1))
        k2 = 1. - (np.log10(t2) - np.log10(teff)) / (np.log10(t2) - np.log10(t1))
    else:
        k1 = 1. - (teff - t1) / (t2 - t1)
        k2 = 1. - (t2 - teff) / (t2 - t1)
    return k1 * rf1 + k2 * rf2


# define a function to regrid on a common wl grid the spectra
def get_interp_spec(data_for_interpolation):
    logg = data_for_interpolation['LogG']
    teff = data_for_interpolation['Teff']
    if data_for_interpolation['exactg']:
        # precise LogG
        if data_for_interpolation['exacttg1']:
            # precise Teff
            wltg = np.copy(data_for_interpolation['wltg1_1'])
            fltg = np.copy(data_for_interpolation['ftg1_1'])
        else:
            # need to interpolate in Teff
            w1 = np.copy(data_for_interpolation['wltg1_1'])
            w2 = np.copy(data_for_interpolation['wltg1_2'])
            f1 = np.copy(data_for_interpolation['ftg1_1'])
            f2 = np.copy(data_for_interpolation['ftg1_2'])
            t1 = np.copy(data_for_interpolation['tg1_1'])
            t2 = np.copy(data_for_interpolation['tg1_2'])
            wltg = get_wlgrid(w1, w2)
            fltg = get_tinterp(wltg, teff, w1, w2, f1, f2, t1, t2, logint=False)
    else:
        # need to interpolate between the two LogG
        g1 = np.copy(data_for_interpolation['g1'])
        g2 = np.copy(data_for_interpolation['g2'])
        if data_for_interpolation['exacttg1']:
            # precise Teff for g1
            wltg1 = np.copy(data_for_interpolation['wltg1_1'])
        else:
            # need to interpolate in Teff
            w1g1 = np.copy(data_for_interpolation['wltg1_1'])
            w2g1 = np.copy(data_for_interpolation['wltg1_2'])
            f1g1 = np.copy(data_for_interpolation['ftg1_1'])
            f2g1 = np.copy(data_for_interpolation['ftg1_2'])
            t1g1 = np.copy(data_for_interpolation['tg1_1'])
            t2g1 = np.copy(data_for_interpolation['tg1_2'])
            wltg1 = get_wlgrid(w1g1, w2g1)
        if data_for_interpolation['exacttg2']:
            # precise Teff for g2
            wltg2 = np.copy(data_for_interpolation['wltg2_1'])
        else:
            # need to interpolate in Teff
            w1g2 = np.copy(data_for_interpolation['wltg2_1'])
            w2g2 = np.copy(data_for_interpolation['wltg2_2'])
            f1g2 = np.copy(data_for_interpolation['ftg2_1'])
            f2g2 = np.copy(data_for_interpolation['ftg2_2'])
            t1g2 = np.copy(data_for_interpolation['tg2_1'])
            t2g2 = np.copy(data_for_interpolation['tg2_2'])
            wltg2 = get_wlgrid(w1g2, w2g2)
        wltg = get_wlgrid(wltg1, wltg2)
        if data_for_interpolation['exacttg1']:
            fltg1 = np.copy(data_for_interpolation['ftg1_1'])
        else:
            fltg1 = get_tinterp(wltg, teff, w1g1, w2g1, f1g1, f2g1, t1g1, t2g1, logint=False)
        if data_for_interpolation['exacttg2']:
            fltg2 = np.copy(data_for_interpolation['ftg2_1'])
        else:
            fltg2 = get_tinterp(wltg, teff, w1g2, w2g2, f1g2, f2g2, t1g2, t2g2, logint=False)
        fltg = get_tinterp(wltg, logg, wltg1, wltg2, fltg1, fltg2, g1, g2, logint=False)
        # print('logg={0} g1={1} g2={2}'.format(logg,g1,g2))
    return wltg, fltg


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