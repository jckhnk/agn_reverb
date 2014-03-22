##agn_reverb

###Flux Redistribution / Random Subset Selection (FR/RSS)

A Monte Carlo simulation approach to calculating the maximal
lag parameter with a corresponding uncertainty from two flux
time series with known measurement uncertainty estimates, which
can then be used to constrain physical models of Active Galactic
Nuclei (AGN) structure in AGN reverberation mapping studies.

The result is the light travel time from the two emitting
regions around the AGN corresponding to the two wavelengths of 
the input flux time series.

The specific procedure is described in detail by Brad Peterson 
[here](http://ned.ipac.caltech.edu/level5/March11/Peterson/peterson2.pdf).

Below are plots of Spitzer and Kepler observations of the AGN zw229,
their cross-correlation function, and an example FR/RSS result.

![](https://github.com/jckhnk/agn_reverb/blob/master/spz_kep_ccf.png?raw=true)

![](https://github.com/jckhnk/agn_reverb/blob/master/tau_centroid_epoch1_sigma_scale1_n1000.png?raw=true)
