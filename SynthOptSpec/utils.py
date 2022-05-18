#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np


def get_spec_file(teff,LogG,modspecdir='Models/bt-settl/'):
    """
    This function takes effective temperature (teff) and Log(g) (LogG) for the synthetic spectra
    and returns the filename for the spectrum. The directory where the synthetic spectra reside 
    is also a parameter.
    """
    teffstr = str(int(teff/100))
    if teff<1000.:
        teffstr = '00'+teffstr
    elif teff<10000.:
        teffstr = '0'+teffstr
    Z = '0.0'
    if teff>2500.:
        specfile = modspecdir+'lte'+teffstr+'-'+LogG+'-'+Z+'a+0.0.BT-NextGen.7.dat.txt'
    else:
        specfile = modspecdir+'lte'+teffstr+'-'+LogG+'-'+Z+'.BT-Settl.7.dat.txt'
    return specfile

def get_g_t_idx(df,myg,myt):
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
        idxg = np.searchsorted(lib_logg,myg,'left')
        if len(myg_idx[0]) == 0:
            exactg = False
            idxg = np.searchsorted(lib_logg,myg,'left')
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
