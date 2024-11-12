# vsdm

By Benjamin Lillard

**Vector space dark matter rate calculation** 


[![arXiv](https://img.shields.io/badge/arXiv-2310.01480%20-green.svg)](https://arxiv.org/abs/2310.01480)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01483%20-green.svg)](https://arxiv.org/abs/2310.01483)


### DESCRIPTION: ##########################################################

VSDM is the Python implementation of the wavelet-harmonic integration method, designed for the efficient calculation of dark matter direct detection scattering rates in anisotropic detectors, and for arbitrary dark matter velocity distributions.

This version introduces an adaptive integration method for projecting 3d functions onto the wavelet-harmonic basis, based on the "wavelet extrapolation" identified in arXiv:2310.01483. The new ExtrapolateFnlm routine uses a polynomial approximation (at linear, cubic, or 7th order) to predict the next generation of wavelet coefficients. In the "refining" stage of the calculation, ExtrapolateFnlm selectively evaluates additional wavelet coefficients until the predictions from the local polynomial expansions match the results from numerical integration everywhere in the space, within some specified precision goal.  

The spherical harmonic functions are also improved in this version. The new normalized associated Legendre function in utilities.py uses just-in-time compilation and an iterative numerical method to gain a factor of 20-25 in speed compared to v0.1, while permitting precise calculations at much larger values of (l, m). Accuracy has been verified out to l=1,000,000.



