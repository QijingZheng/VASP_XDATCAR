#!/usr/bin/env python
# This script was created by zqj
# <Tue May 26 09:05:25 CST 2015>

##################################### NOTES ##################################### 
# 1. Set NBLOCK = 1 in the INCAR, so that all the configuration is wrtten to
# XDATCAR.
################################################################################# 

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, fftshift

# Boltzmann Constant in [eV/K]
kb = 8.617332478E-5
# electron volt in [Joule]
ev = 1.60217733E-19
# Avogadro's Constant
Navogadro = 6.0221412927E23

################################################################################      
class xdatcar:
    """ Python Class for VASP XDATCAR """

    def __init__(self, File=None):
        if File is None:
            self.xdatcar = 'XDATCAR'
        else:
            self.xdatcar = File

        # time step of MD
        self.potim = None
        # mass per type
        self.mtype = None
        self.readoutcar()

        self.TypeName = None
        self.ChemSymb = None
        self.Ntype = None
        self.Nions = None
        self.Nelem = None
        self.Niter = None

        # position in Direct Coordinate
        self.position = None
        # position in Cartesian Coordinate
        self.positionC = None
        # Velocity in Angstrom per Femtosecond
        self.velocity = None
        self.readxdat()

        self.mass_and_name_per_ion()
        # Temperature
        self.Temp = np.zeros(self.Niter-1)
        # Kinetic Energy
        self.Ken = np.zeros(self.Niter-1)
        # Time in femtosecond
        self.Time = np.arange(self.Niter-1) * self.potim
        self.getTemp()

        # Velocity Autocorrelation Function 
        self.VAF = None
        self.VAF2= None
        # Pair Correlation Function
        # self.PCF = None

    def mass_and_name_per_ion(self):
        # mass per ion
        self.mions = []
        self.ChemSymb = []

        if self.TypeName is None:
            self.TypeName = [chr(i) for i in range(65,91)][:self.Ntype]

        for i in range(self.Ntype):
            self.mions += [np.tile(self.mtype[i], self.Nelem[i])]
            self.ChemSymb += [np.tile(self.TypeName[i], self.Nelem[i])]

        self.mions = np.concatenate(self.mions)
        self.ChemSymb = np.concatenate(self.ChemSymb)

    def readxdat(self):
        """ Read VASP XDATCAR """

        # inp = [line for line in open(self.xdatcar) if line.strip()]
        inp = open(self.xdatcar).readlines()

        scale      = float(inp[1])
        self.cell  = np.array([line.split() for line in inp[2:5]], dtype=float)
        self.cell *= scale

        ta = inp[5].split()
        tb = inp[6].split()

        if ta[0].isalpha():
            self.TypeName = ta
            self.Ntype    = len(ta)
            self.Nelem    = np.array(tb, dtype=int)
            self.Nions    = self.Nelem.sum()
            self._nhead   = 8
        else:
            # Names of each elements not written in XDATCAR head
            self.Nelem    = np.array(ta, type=int)
            self.Nions    = self.Nelem.sum()
            self.Ntype    = len(ta)
            self.TypeName = None
            self._nhead   = 7

        # For ISIF >= 3, VASP stores cell shapes at each step
        if self.isif >= 3:
            # No. of iterations
            self.Niter = len(inp) // (self._nhead + self.Nions)
            if len(inp) % (self._nhead + self.Nions) != 0:
                raise ValueError("XDATCAR may have been corrupted!")

            self.position  = np.array(
                [
                    [line.split() for line in inp[
                        self._nhead + ii * (self.Nions + self._nhead)
                        :
                        self._nhead + ii * (self.Nions + self._nhead) + self.Nions
                        ]
                    ]
                    for ii in range(self.Niter)
                ], dtype=float
            )

            self.scales = np.array([
                    inp[ii*(self.Nions + self._nhead)+1] for ii in range(self.Niter)
                ],
                dtype=float
            )

            self.cells = np.array(
                [
                    [line.split() for line in inp[
                        2 + ii * (self.Nions + self._nhead)
                        :
                        2 + ii * (self.Nions + self._nhead) + 3
                        ]
                    ]
                    for ii in range(self.Niter)
                ], dtype=float
             ) * self.scales[:,None,None]

            self.positionC = np.zeros_like(self.position)
            for ii in range(self.Niter):
                self.positionC[ii,:,:] = np.dot(self.position[ii,:,:], self.cells[ii])
                
        else:
            # No. of iterations
            self.Niter = (len(inp) - self._nhead - 1) // (1 + self.Nions)
            if (len(inp) - self._nhead - 1) % (1 + self.Nions) != 0:
                raise ValueError("XDATCAR may have been corrupted!")

            self.position  = np.array(
                [
                    [line.split() for line in inp[
                        self._nhead + ii * (self.Nions+1)
                        :
                        self._nhead + ii * (self.Nions+1) + self.Nions
                        ]
                    ]
                    for ii in range(self.Niter)
                ], dtype=float
            )
            self.positionC = np.tensordot(self.position, self.cell, axes=(2,0))

            # Velocity is ill-defined for varied-shape cell
            dpos = np.diff(self.position, axis=0)
            # apply periodic boundary condition
            dpos[dpos > 0.5] -= 1.0
            dpos[dpos <-0.5] += 1.0
            # Velocity in Angstrom per femtosecond
            for i in range(self.Niter-1):
                dpos[i,:,:] = np.dot(dpos[i,:,:], self.cell) / self.potim

            self.velocity = dpos


    def readoutcar(self):
        """ read POTIM and POMASS from OUTCAR """

        if os.path.isfile("OUTCAR"):
            # print "OUTCAR found!"
            # print "Reading POTIM & POMASS from OUTCAR..."
            
            outcar = [line.strip() for line in open('OUTCAR')]
            lm = 0;
            for ll, line in enumerate(outcar):
                if 'POTIM  =' in line:
                    # lp = ll
                    self.potim = float(line.split()[2])

                # For ISIF >= 3, VASP output CELLs for each step
                if 'ISIF   =' in line:
                    self.isif = int(line.split()[2])

                if 'Mass of Ions in am' in line:
                    lm = ll + 1

                if lm:
                    break

            # In case Masses not written in OUTCAR
            if lm == 0:
                raise ValueError("Masses for atoms NOT found! Check OUTCAR to see if 'POMASS' for atoms are present!")

            # For heavy atoms, digits for atomic masses may stick together,
            # resulting in cases like: "POMASS =  95.94 32.07183.85"

            pomass_line = outcar[lm]
            pomass_tmp  = pomass_line.split()[2:]
            # Count the number of decimal points, which should equal to the
            # number of types of elements
            if len(pomass_tmp) != pomass_line.count('.'):
                # Fortunately, VASP use fixed-format for printing the atomic
                # masses, i.e. the number of decimal digits for all the floats
                # are the same. Check the last float for this number.

                nd   = pomass_line[::-1].index('.')
                # Find the positions for the decimal points, and add "nd"
                dpos = [ii+nd for ii,xx in enumerate(pomass_line) if xx == '.']
                # Add extra space to the end the number and rejoin the string
                pomass_new_line = ''.join(
                        [xx + ' ' if ii in dpos
                         else xx
                         for ii, xx in
                         enumerate(pomass_line)]
                    )
                self.mtype = np.array(pomass_new_line.split()[2:], dtype=float)
            else:
                self.mtype = np.array(pomass_tmp, dtype=float)

    def getTemp(self, Nfree=None):
        """ Temp vs Time """

        for i in range(self.Niter-1):
            ke = np.sum(np.sum(self.velocity[i,:,:]**2, axis=1) * self.mions / 2.)
            self.Ken[i] = ke * 1E7 / Navogadro / ev
            if Nfree is None:
                Nfree = 3 * (self.Nions - 1)
            self.Temp[i] = 2 * self.Ken[i] / (kb * Nfree)

    def getVAF(self):
        """ Velocity Autocorrelation Function """

        # VAF definitions
        # VAF(t) = Natoms^-1 * \sum_i <V_i(0) V_i(t)>
        ############################################################
        # Fast Fourier Transform Method to calculate VAF
        ############################################################
        # The cross-correlation theorem for the two-sided correlation:
        # corr(a,b) = ifft(fft(a)*fft(b).conj()

        # If a == b, then this reduces to the special case of the
        # Wiener-Khinchin theorem (autocorrelation of a):

        # corr(a,a) = ifft(abs(fft(a))**2)
        # where the power spectrum of a is simply:
        # fft(corr(a,a)) == abs(fft(a))**2
        ############################################################
        # in this function, numpy.correlate is used to calculate the VAF

        self.VAF2 = np.zeros((self.Niter-1)*2 - 1)
        for i in range(self.Nions):
            for j in range(3):
                self.VAF2 += np.correlate(self.velocity[:,i,j],
                                          self.velocity[:,i,j], 
                                          'full')
        # two-sided VAF
        self.VAF2 /=  np.sum(self.velocity**2)
        self.VAF = self.VAF2[self.Niter-2:]

    def phononDos(self, unit='THz', sigma=5):
        """ Phonon DOS from VAF """

        N = self.Niter - 1
        # Frequency in THz
        omega = fftfreq(2*N-1, self.potim) * 1E3
        # Frequency in cm^-1
        if unit.lower() == 'cm-1':
            omega *= 33.35640951981521
        if unit.lower() == 'mev':
            omega *= 4.13567
        # from scipy.ndimage.filters import  gaussian_filter1d as gaussian
        # smVAF = gaussian(self.VAF2, sigma=sigma)
        # pdos = np.abs(fft(smVAF))**2
        if self.VAF2 is None:
            self.getVAF()
        pdos = np.abs(fft(self.VAF2 - np.average(self.VAF2)))**2

        return omega[:N], pdos[:N]

    def PCF(self, bins=50, Niter=10, A='', B=''):
        """ Pair Correlation Function """

        if not A:
            A = self.TypeName[0]
        if not B:
            B = A

        whichA = self.ChemSymb == A
        whichB = self.ChemSymb == B
        indexA = np.arange(self.Nions)[whichA]
        indexB = np.arange(self.Nions)[whichB]
        posA = self.position[:,whichA,:]
        posB = self.position[:,whichB,:]

        steps = range(0, self.Niter, Niter)
        rABs = np.array([posA[i,k,:]-posB[i,j,:]
                         for k in range(indexA.size)
                         for j in range(indexB.size)
                         for i in steps
                         if indexA[k] != indexB[j]])
        # periodic boundary condition
        rABs[rABs > 0.5] -= 1.0
        rABs[rABs <-0.5] += 1.0
        # from direct to cartesian coordinate
        rABs = np.linalg.norm(np.dot(self.cell, rABs.T), axis=0)
        # histogram of pair distances
        val, b = np.histogram(rABs, bins=bins)
        # density of the system
        rho = self.Nions / np.linalg.det(self.cell)
        # Number of A type atom
        Na = self.Nelem[self.TypeName.index(A)]
        # Number of B type atom
        Nb = self.Nelem[self.TypeName.index(B)]
        dr = b[1] - b[0]
        val = val * self.Nions / (4*np.pi*b[1:]**2 * dr) / (Na * Nb * rho) / len(steps)

        return val, b[1:]

################################################################################      

# test code of the above class
if __name__ == '__main__':
    inp = xdatcar()
    inp.getVAF()
    # plt.plot((np.abs(fft(inp.VAF[inp.Niter-2:]))**2))
    # print inp.VAF.shape
    # plt.plot(inp.Time, inp.VAF, 'ko-', lw=1.0, ms=2,
    #         markeredgecolor='r', markerfacecolor='red')
    # 
    # plt.xlabel('Time [fs]')
    # plt.ylabel('Velocity Autocorrelation Function')

    # x, y = inp.phononDos('cm-1')
    # plt.plot(x, y, 'ko-')
    # plt.xlim(0, 5000)
    # # plt.ylim(-0.5, 1.0)

    val, b = inp.PCF(100, 1)
    plt.plot(b, val)
    plt.axhline(y=1, color='r')
    plt.xlim(0, 5)
    plt.show()
