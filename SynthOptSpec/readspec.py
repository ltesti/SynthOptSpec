#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
                        
import numpy as np
#import scipy.interpolate as ssi
#import matplotlib.pyplot as plt
#import os
from astropy.table import Table

from .get_spec_file import nrefrac


def read_phoenix_txt(infile, wlextrmin, wlextrmax, modflux_log, correct_vacuum):
    """
    """
    print('Opening file: {0}'.format(infile))
    f = open(infile, 'r')
    wl = []
    fl = []
    for line in f:
        line = line.strip()
        columns = line.split()
        if columns[0] != '#':
            mywl = float(columns[0].replace("D", "E"))
            if (mywl >= wlextrmin) & (mywl <= wlextrmax):

                myf = float(columns[1].replace("D", "E"))
                try :
                    wl.append(mywl)
                    if modflux_log:
                        fl.append(10 ** myf)
                    else:
                        fl.append(myf)
                except OverflowError:
                    print(f'Error (Overflow) myf={myf}')
    f.close()
    vac_wl = np.array(wl, dtype=float)
    if correct_vacuum:
        read_wl = vac_wl / (1 + 1.e-6 * nrefrac(vac_wl))
    else:
        read_wl = np.copy(vac_wl)
    read_fl = np.array(fl, dtype=float)

    return read_wl, read_fl

def read_phoenix_fits(infile, wlextrmin, wlextrmax, correct_vacuum):
    """
    """
    spt = Table.read(infile, hdu=1)
    wl = 10000. * np.array(spt['Wavelength'], dtype=float)
    ng = np.where((wl >= wlextrmin) & (wl <= wlextrmax))
    vac_wl = wl[ng]
    if correct_vacuum:
        read_wl = vac_wl / (1 + 1.e-6 * nrefrac(vac_wl))
    else:
        read_wl = np.copy(vac_wl)
    read_fl = np.array(spt['Flux'], dtype=float)[ng]

    return read_wl, read_fl

def read_atlas9(infile, wlextrmin, wlextrmax, correct_vacuum=False):
    """
    """
    f = open(infile, 'r')
    wl = []
    fl = []
    for line in f:
        line = line.strip()
        columns = line.split()
        if columns[0] != '#':
            mywl = float(columns[2].replace("D", "E"))
            if (mywl >= wlextrmin) & (mywl <= wlextrmax):

                myf = float(columns[4].replace("D", "E"))
                try :
                    wl.append(mywl)
                    fl.append(myf)
                except OverflowError:
                    print(f'Error (Overflow) myf={myf}')
    f.close()
    vac_wl = 10.*np.array(wl, dtype=float)
    if correct_vacuum:
        read_wl = vac_wl / (1 + 1.e-6 * nrefrac(vac_wl))
    else:
        read_wl = np.copy(vac_wl)
    read_fl = 4.e18*np.array(fl, dtype=float)*2.998e+8/read_wl/read_wl

    return read_wl, read_fl