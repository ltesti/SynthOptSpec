#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import types

from .extinction import cardelli_extinction
from .resample import resamp_spec
#from .get_spec_file import get_spec_file

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

def get_phot_spec(df, logg, teff, logint_flux=False, logint_pars=False):
    # gets the spectra from the library and interpolates at the resolution
    # of the lower resolution spectrum in the (up to) four spectra
    #
    data_for_interpolation = get_g_t_idx(df, logg, teff)
    wl, fl = get_interp_spec(data_for_interpolation,logint_flux=logint_flux, logint_pars=logint_pars)
    #
    return wl, fl

def apply_rvel(wl, rvel):
    #
    ckms = 299792.458  # speed of light in km/s
    # Radial velocity
    wl = wl * (1. + rvel / ckms)
    #
    return wl

def get_spec(df, logg, teff, wlmin=4750.1572265625, wlmax=9351.4072265625, dl=1.25,
             av=None, rv=3.1, rvel=None, normalization="Dominika", logint_flux=False, logint_pars=False):
    '''
    Optional parameters:
    rvel : radial velocity
    '''
    #
    # gets the spectra from the library and interpolates at the resolution
    # of the lower resolution spectrum in the (up to) four spectra
    #
    wl, fl = get_phot_spec(df, logg, teff, logint_flux=logint_flux, logint_pars=logint_pars)

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


def get_tinterp(wltg, teff, w1, w2, f1, f2, t1, t2, logint_flux=False, logint_pars=False):
    #
    rf1 = resamp_spec(wltg, w1, f1)
    rf2 = resamp_spec(wltg, w2, f2)
    if logint_pars:
        k1 = 1. - (np.log10(teff) - np.log10(t1)) / (np.log10(t2) - np.log10(t1))
        k2 = 1. - (np.log10(t2) - np.log10(teff)) / (np.log10(t2) - np.log10(t1))
    else:
        k1 = 1. - (teff - t1) / (t2 - t1)
        k2 = 1. - (t2 - teff) / (t2 - t1)
    if logint_flux:
        rf_int = 10**(k1 * np.log10(rf1) + k2 * np.log10(rf2))
    else:
        rf_int = k1 * rf1 + k2 * rf2
    return rf_int


# define a function to regrid on a common wl grid the spectra
def get_interp_spec(data_for_interpolation, logint_flux=False, logint_pars=False):
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
            fltg = get_tinterp(wltg, teff, w1, w2, f1, f2, t1, t2, logint_flux=logint_flux, logint_pars=logint_pars)
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
            fltg1 = get_tinterp(wltg, teff, w1g1, w2g1, f1g1, f2g1, t1g1, t2g1, logint_flux=logint_flux)
        if data_for_interpolation['exacttg2']:
            fltg2 = np.copy(data_for_interpolation['ftg2_1'])
        else:
            fltg2 = get_tinterp(wltg, teff, w1g2, w2g2, f1g2, f2g2, t1g2, t2g2, logint_flux=logint_flux)
        fltg = get_tinterp(wltg, logg, wltg1, wltg2, fltg1, fltg2, g1, g2, logint_flux=logint_flux)
        # print('logg={0} g1={1} g2={2}'.format(logg,g1,g2))
    return wltg, fltg
