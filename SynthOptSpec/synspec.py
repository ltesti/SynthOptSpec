#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
                        
import numpy as np
import scipy.interpolate as ssi
import matplotlib.pyplot as plt
import os
from astropy.table import Table

from .utils import resamp_spec, nrefrac
from .accel.utils_c import resamp_spec_c, smooth_c
from .compute_magnitude import ComputeMag, read_standard_filters

class SynSpec(object):
    """
        The class reads a Synthetic Spectrum from a file and provides methods to smooth it
        params is a dictionary that contains the input parameters
        
        params:
           'file' : full path to the input file
           'wlmin' : minimum wavelength for processing
           'wlmax' : maximum wavelength for processing
           'wledge' : optional parameter to set an extra wavelength margin when extracting the spectrum
        """
    
    def __init__(self, parameters):
        """
        set up the object: this one reads the spectrum within given boundaries
        """
                
        self.params = parameters
        
        self.infile = self.params['file']
        self.modflux_log = self.params['modflux_log']
        if 'format' in self.params.keys():
            self.file_format = self.params['format']
        else:
            self.file_format = 'fits'
        self.wlmin = self.params['wlmin']
        self.wlmax = self.params['wlmax']
        if 'correct_vacuum' in self.params.keys():
            self.correct_vacuum = self.params.keys()
        else:
            self.correct_vacuum = True
        
        # set minimum and maximum wavelengths to extract spectra
        self.wlextrmin, self.wlextrmax = self._set_readedges()
        
        # read spectrum, full resolution, within wavelength boundaries
        self.aswl, self.asfl = self.getspec()
        self.nwl = np.where((self.aswl>=self.wlmin) & (self.aswl<=self.wlmax))
        self.swl = self.aswl[self.nwl]
        self.sfl = self.asfl[self.nwl]

        #
        self.has_filters = False
        
    def _set_readedges(self):
        """
        method to define the edges of the extracted spectrum
        """

        if 'wledge' in self.params.keys():
            self.wledge = self.params['wledge']
        else:
            self.wledge = 0.0
        #
        return self.wlmin-self.wledge, self.wlmax+self.wledge
    
    def getspec(self):
        """
        method to read the spectrum within the defined wavelength boundaries
        """

        if (self.file_format == 'old') or (self.file_format == 'txt'):
            f = open(self.infile, 'r')
            wl = []
            fl = []
            for line in f:
                line = line.strip()
                columns = line.split()
                if columns[0] != '#':
                    mywl = float(columns[0].replace("D", "E"))
                    if (mywl >= self.wlextrmin) & (mywl <= self.wlextrmax):

                        myf = float(columns[1].replace("D", "E"))
                        try :
                            wl.append(mywl)
                            if self.modflux_log:
                                fl.append(10 ** myf)
                            else:
                                fl.append(myf)
                        except OverflowError:
                            print(f'Error (Overflow) myf={myf}')
            f.close()
            vac_wl = np.array(wl, dtype=float)
            if self.correct_vacuum:
                read_wl = vac_wl / (1 + 1.e-6 * nrefrac(vac_wl))
            else:
                read_wl = np.copy(vac_wl)
            read_fl = np.array(fl, dtype=float)
        else:
            spt = Table.read(self.infile, hdu=1)
            wl = 10000. * np.array(spt['Wavelength'], dtype=float)
            ng = np.where((wl >= self.wlextrmin) & (wl <= self.wlextrmax))
            vac_wl = wl[ng]
            if self.correct_vacuum:
                read_wl = vac_wl / (1 + 1.e-6 * nrefrac(vac_wl))
            else:
                read_wl = np.copy(vac_wl)
            read_fl = np.array(spt['Flux'], dtype=float)[ng]
        #
        nsort = np.argsort(read_wl)
        return read_wl[nsort], read_fl[nsort]
    
    def plotspec(self, smoothed=False, resampled=True, outfile=None, showedge=True):
        #
        fig = plt.figure(figsize=(14,7))
        #
        #
        if showedge:
            plt.plot(self.aswl,self.asfl,color='orange',alpha=0.3)
            plt.xlim(self.wlextrmin,self.wlextrmax)
        else:
            plt.xlim(self.wlmin,self.wlmax)
        plt.plot(self.swl,self.sfl,color='r',alpha=0.5)
        if resampled:
            plt.plot(self.rswl,self.rsfl,color='g',alpha=0.85)
        #
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Fluxm (model units)')
        #
        if smoothed:
            if showedge:
                plt.plot(self.aswl,self.sasfl,color='lightgreen',linestyle='dotted',alpha=0.6) 
            plt.plot(self.swl,self.ssfl,color='royalblue',linestyle='dotted')
            if resampled:
                plt.plot(self.rswl,self.rsfl,color='cyan',linestyle='dashed')
        #
        if outfile:
            plt.savefig(outfile)

    def smoothspec(self, R=4000., set_ssfl_attribute=True, muse_res_manual=True, verbose=False, constantR=True, modelR=None, nsig=5., accel=True):
        """
        method to do a gaussian convolution to reach a final resolution R
        in this initial version, we are assuming that modelR >> R (infinite
        resolution approximation), so that the sigma of the convolution gaussian 
        is the only parameter that defines the final resolution
        
        R = final resolution this has become optional and we use the MUSE table as default
        set_ssfl_attribute = True/False controls whether the smoothed spectrum is stored in the object
                             if set_ssfl_attribute is True, then the method will store the smoothed spectrum    
                             in the attributes self.sasfl and self.ssfl, otherwise the method will return the 
                             full smoothed spectrum (including the possible edges)
        muse_res_manual = True/False use the resolution table provided in the MUSE manual (P110)
        constantR = True/False allows the option of non constant R (not yet implemented)
        modelR = value of the model resolution, to include the intrinsic resolution in the final result
                 (not yet implemented)
        nsig = 5. (number of sigmas for the numerical computation of the gaussian convolution)
        
        """

        if accel:
            ssfl = smooth_c(self.aswl, self.asfl, R, muse_res_manual, verbose, nsig)
        else:
            #
            def fgauss(x,s):
                return np.exp(-x**2/(2.*s**2))

            # These are from the MUSE manual Section 3.2 (P110 version)
            muse_res_wl = np.array([4650.0,5000.0,5500.0,6000.0,6500.0,7000.0,7500.0,8000.0,8500.0,9000.0,9350.0])
            muse_res = np.array([1609.,1750.,1978.,2227.,2484.,2737.,2975.,3183.,3350.,3465.,3506.])

            ssfl = np.zeros(len(self.asfl))
            dl = self.aswl / R
            if muse_res_manual:
                if verbose: print("Using MUSE-res-manual ({0}, {1}) => ".format(dl[0],dl[-1]))
                muse_res_interp = ssi.interp1d(muse_res_wl,muse_res)
                nlow = np.where(self.aswl<=muse_res_wl[0])
                dl[nlow] = self.aswl[nlow]/muse_res[0]
                nhigh = np.where(self.aswl>=muse_res_wl[-1])
                dl[nhigh] = self.aswl[nhigh]/muse_res[-1]
                ninterp = np.where((self.aswl>muse_res_wl[0]) & (self.aswl<muse_res_wl[-1]))
                dl[ninterp] = self.aswl[ninterp]/muse_res_interp(self.aswl[ninterp])
                if verbose: print(" ({0}, {1})\n".format(dl[0], dl[-1]))

            sdl = dl / (2. * np.sqrt(2. * np.log(2.)))

            for i in range(len(self.asfl)):
                #dl = self.aswl[i]/R
                #nsig = 5.
                #sdl = dl/(2.*np.sqrt(2.*np.log(2.)))
                msdl = self.aswl[i]-nsig*sdl[i]
                psdl = self.aswl[i]+nsig*sdl[i]
                nsm = np.where((self.aswl >= msdl) & (self.aswl<=psdl))
                smwl = self.aswl[nsm]
                fs = 0.
                area = 0.
                for j in range(len(smwl)):
                    fg = fgauss(smwl[j]-self.aswl[i],sdl[i])
                    fs += fg*self.asfl[nsm[0][j]]
                    area += fg
                ssfl[i] = fs/area
        #
        if set_ssfl_attribute:
            self.sasfl = ssfl
            self.ssfl = self.sasfl[self.nwl]
        else:
            return ssfl
            
    def resample(self, wlsamp, smoothed=False, set_rsfl_attribute=True, accel=True):
        """
        method to resample the spectrum on a new wavelength grid
        the assumption is to do a simple binning - average flux per wl bin
        """

        if smoothed:
            flvec = np.copy(self.sasfl)
        else:
            flvec = np.copy(self.asfl)
        if set_rsfl_attribute:
            if accel:
                self.rsfl = resamp_spec_c(wlsamp, self.aswl, flvec)
            else:
                self.rsfl = resamp_spec(wlsamp, self.aswl, flvec)
            self.rswl = wlsamp
        else:
            if accel:
                return resamp_spec_c(wlsamp, self.aswl, flvec)
            else:
                return resamp_spec(wlsamp, self.aswl, flvec)

    def spec_to_mag(self):
        """Used to compute magnitudes from spectra

        This function will read the default filters provided in the package AstroFilterTransmissions
        directory and will compute apparent magnitudes from spectra using procedures in compute_nagnitude.py

        Note: the infrared filters have the wavelength scale and f0 in usits of [um] and [W/m2/um]
               respectively
        """
        if not self.has_filters:
            self.wlfilters, self.f0, self.trasp, self.filters = read_standard_filters(sos_units=True)
            self.has_filters = True

        # we pass the full resolution files, note that this is not correct if we
        # introduce a variation of the velocity
        # self.aswl, self.asfl
        # TODO: investigate the idea of inserting an option to pass lower resolution spectra
        self.mags = {}
        self.fluxes = {}
        self.wlmag = {}
        for i in range(len(self.filters)):
            self.mags[self.filters[i]], self.fluxes[self.filters[i]] = ComputeMag(self.aswl, self.asfl,
                                               self.trasp['wl'+self.filters[i]], self.trasp['tr'+self.filters[i]],
                                               self.f0[self.filters[i]])

    