# vsdm

By Benjamin Lillard

**Vector space dark matter rate calculation** 


[![arXiv](https://img.shields.io/badge/arXiv-2310.01480%20-green.svg)](https://arxiv.org/abs/2310.01480)
[![arXiv](https://img.shields.io/badge/arXiv-2310.01483%20-green.svg)](https://arxiv.org/abs/2310.01483)


### DESCRIPTION: ##########################################################

VSDM is the Python implementation of the wavelet-harmonic integration method, designed for the efficient calculation of dark matter direct detection scattering rates in anisotropic detectors, and for arbitrary dark matter velocity distributions. Each input function is projected onto a basis of orthogonal functions (spherical harmonics and wavelets), so that the scattering rate calculation becomes a linear operation on the vector space representations of the functions. This method is introduced in arXiv:2310.01480, with the relevant details worked out in arXiv:2310.01483. 

Version 0.3 of this code introduces an adaptive integration method for projecting 3d functions onto the wavelet-harmonic basis, based on the "wavelet extrapolation" identified in arXiv:2310.01483. The new AdaptiveFn and WaveletFnlm routines use a polynomial approximation (at linear, cubic, or 7th order) to predict the next generation of wavelet coefficients. In the "refining" stage of the calculation, WaveletFnlm selectively evaluates additional wavelet coefficients until the predictions from the local polynomial expansions match the results from numerical integration everywhere in the space, within some specified precision goal.  

The spherical harmonic functions are also improved in this version. The new normalized associated Legendre function in utilities.py uses just-in-time compilation and an iterative numerical method to gain a factor of 20-25 in speed compared to v0.1, while permitting precise calculations at much larger values of (l, m). Accuracy has been verified out to l=1,000,000.

A few explanatory notebooks are included in the 'tools' directory:
- **Calculating Coefficients:** demonstrates how to calculate the wavelet-harmonic coefficients using the Fnlm and EvaluateFnlm classes, and how to import saved coefficients from a csv file. 
- **Rate Calculation:** provides a few examples of how to perform the rate calculation for arbitrary detector orientations with respect to the dark matter velocity distribution. 
-**Wigner D and G**: a brief introduction to the Wigner $D^{(\ell)}$ and $G^{(\ell)}$ matrices, which encode the action of rotations on complex or real spherical harmonics (respectively). 
The 'tools' directory also includes three sample Python codes:
- **demo_fs2.py**: uses the WaveletFnlm method to calculate batches of wavelet-harmonic coefficients $\langle f_s^2 | n \ell m \rangle$, for the particle-in-a-box momentum form factors $f_s^2(\vec q$) used in the demonstration notebooks and arXiv:2310.01483. 
-**demo_gX.py**: calculates $\langle g_\chi | n \ell m \rangle$ for a velocity distribution example $g_\chi(\vec v$), defined as the sum of four Gaussians with different average velocities and dispersions. This is the velocity distribution used in "Calculating Coefficients" and "Rate Calculation". 
-**SHM_gX.py**: this tool calculates the wavelet-harmonic expansion for the Standard Halo Model (SHM) velocity distribution, as a function of galactic frame Earth velocity, using the adaptive WaveletFnlm method. 
Four CSV files are included in the 'demo' directory: 
-'demo_fs2.csv' and 'demo_fs2_alt.csv', with the values of $\langle f_s^2 | n \ell m \rangle$ for the $\vec n = (1, 1, 2)$ and $\vec n = (3, 2, 1)$ excited states, respectively, of the particle-in-a-box model. 
-'gX_model4.csv': $\langle g_\chi | n \ell m \rangle$ for the four-gaussian velocity distribution.
-'SHM_v250.csv': $\langle g_\chi | n \ell m \rangle$ for the Standard Halo Model, with a dark matter wind speed of 250 km/s. The galactic escape velocity and local group circular speed are set to 544 km/s and 238 km/s, respectively. 

