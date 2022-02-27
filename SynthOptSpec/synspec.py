#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
                        
import numpy as np
import matplotlib.pyplot as plt


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
        self.wlmin = self.params['wlmin']
        self.wlmax = self.params['wlmax']
        
        # set minimum and maximum wavelengths to extract spectra
        self.wlextrmin, self.wlextrmax = self._set_readedges()
        
        # read spectrum, full resolution, within wavelength boundaries
        self.aswl, self.asfl = self.getspec()
        self.nwl = np.where((self.aswl>=self.wlmin) & (self.aswl<=self.wlmax))
        self.swl = self.aswl[self.nwl]
        self.sfl = self.asfl[self.nwl]
        
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
        f = open(self.infile, 'r')
        wl = []
        fl = []
        for line in f:
            line = line.strip()
            columns = line.split()
            if columns[0] != '#':
                mywl = columns[0]
                if ((float(mywl) >= self.wlextrmin) & (float(mywl) <= self.wlextrmax)):
                    wl.append(mywl)
                    fl.append(columns[1]) 
        f.close()
        # aswl = np.array(wl,dtype=float)
        # asfl = np.array(fl,dtype=float)
        #
        # nwl = np.where((aswl >= self.wlextrmin) & (aswl <= self.wlextrmax))
        #
        # return aswl[nwl], asfl[nwl]
        return np.array(wl,dtype=float), np.array(fl,dtype=float)
    
    def plotspec(self, smoothed=False, outfile=None, showedge=True):
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
        #
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Fluxm (model units)')
        #
        if smoothed:
            if showedge:
                plt.plot(self.aswl,self.sasfl,color='lightgreen',linestyle='dotted',alpha=0.6) 
            plt.plot(self.swl,self.ssfl,color='royalblue',linestyle='dotted')
        #
        if outfile:
            plt.savefig(outfile)

    def smoothspec(self, R, set_ssfl_attribute=True, constantR=True, modelR=None, nsig=5.):
        """
        method to do a gaussian convolution to reach a final resolution R
        in this initial version, we are assuming that modelR >> R (infinite
        resolution approximation), so that the sigma of the convolution gaussian 
        is the only parameter that defines the final resolution
        
        R = final resolution (in the initial version is assumed to be constant)
        set_ssfl_attribute = True/False controls whether the smoothed spectrum is stored in the object
                             if set_ssfl_attribute is True, then the method will store the smoothed spectrum    
                             in the attributes self.sasfl and self.ssfl, otherwise the method will return the 
                             full smoothed spectrum (including the possible edges)
        constantR = True/False allows the option of non constant R (not yet implemented)
        modelR = value of the model resolution, to include the intrinsic resolution in the final result
                 (not yet implemented)
        nsig = 5. (number of sigmas for the numerical computation of the gaussian convolution)
        
        """
        #
        def fgauss(x,s):
            return np.exp(-x**2/(2.*s**2))

        ssfl = np.zeros(len(self.asfl))
        for i in range(len(self.asfl)):
            dl = self.aswl[i]/R
            #nsig = 5.
            sdl = dl/(2.*np.sqrt(2.*np.log(2.)))
            msdl = self.aswl[i]-nsig*sdl
            psdl = self.aswl[i]+nsig*sdl
            nsm = np.where((self.aswl >= msdl) & (self.aswl<=psdl))
            smwl = self.aswl[nsm]
            fs = 0.
            area = 0.
            for j in range(len(smwl)):
                fg = fgauss(smwl[j]-self.aswl[i],sdl)
                fs += fg*self.asfl[nsm[0][j]]
                area += fg
            ssfl[i] = fs/area
        #
        if set_ssfl_attribute:
            self.sasfl = ssfl
            self.ssfl = self.sasfl[self.nwl]
        else:
            return ssfl
    