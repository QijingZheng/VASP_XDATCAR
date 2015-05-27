# VASP_XDATCAR
A python class for parsing VASP XDATCAR from molecular dynamics.

The XDATCAR file contains the trajectory during a molecular dynamics run, i.e. the positions of all the atoms at each time step.
From this information, we may calculate the following physical quantity

1. the time-dependent temperature of the system
2. Velocity Autocorrelation Function (VAF) and Phonon Density of States
3. Pair Correlation Function (PCF)
