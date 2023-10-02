# vsdm
Vector space dark matter rate calculation

Version 0.1.0



### DESCRIPTION: ##########################################################

vsdm/
	pyproject.toml
	README.md
	tools/
		demo/
			fs2_box_4_7_10.csv
			gX_model4.csv
		Plots_Angular.ipynb
		Plots_Evaluation.ipynb
		Plots_fs2.ipynb
		Plots_Radial.ipynb
		Plots_RateCalc.ipynb
	vsdm/
		__init__.py
		units.py
		utilities.py
		analytic.py
		basis.py
		gaussians.py
		projection.py
		matrixcalc.py
		portfolio.py
		wigner.py
		ratecalc.py

units.py includes 'eV', 'keV', 'MeV'; also electron mass, inverse Bohr radius qBohr
	Units are hbar=1; units of [energy] and [velocity] can be set by user.
	Default: [energy] = 1*eV, [velocity] = c.
Unless the user has some units package of their own, I recommend loading the module with:
	from vsdm.units import *
	import vsdm
This way, can consistently and conveniently use 'eV', 'km_s' units.

__init.py__ set up to import all submodules into vsdm namespace.
	e.g. vsdm.ProjectFnlm, rather than vsdm.projection.ProjectFnlm.


####	SAVED FILE TYPES:

ProjectFnlm has a CSV method, one row per (nlm) coefficient.
* Rationale: <f|nlm> are simple vector objects, sometimes known analytically.
	Can use CSV to switch between Mathematica and Python easily, for example.
* Pro: CSV files are more human-readable
* Con: If a project requires large numbers of gX or fgs2 functions,
	may want to assemble hdf5 databases to store them.
ProjectFnlm can also save to hdf5 database.

MathcalI uses hdf5, to store 3d array (ell, nv, nq).
* Rationale: MathcalI generates and uses large 3d arrays, hdf5 is obvious choice
* Con: 'none'. 3d arrays are not very readable in the first place.

####  HDF5 DATABASE.

Within a specific hdf5 file (e.g. analysisName.hdf5), there should be
	ONE basis V and ONE basis Q. These should match between gX, mcalI, and fsg2.

Each type of object (<gX|V>, <V|I|Q>, <Q|fgs2>) has multiple instances,
	labeled by an index; 'gX_ix', 'fs2_ix', 'mI_ix'.
Part of the mcalI index is associated with part of the fgs2 index: that is,
	the final state label 's', 'g -> s', determines the excitation energy 'omegaS'

An analysis depends on...
	(gX_i), the DM velocity model;
	(fs_j), the model for the |f_gs|^2 form factor;
	(mX, F_DM), the DM particle model (including SM-DM form factor)
	(rotation), the orientation of the detector
If there are many items in each of these four lists,
	then project each object onto the chosen (V,Q) basis.
The event rate is found by matrix multiplication.

To optimize the rotation calculation, we define a mcalK matrix
	by summing over the radial modes for each value of ell.
This is done for every mK_ix = (gX_i, fs_i, mX, F_DM).
Evaluating the mcalK matrices from mcalI is usually fast, 
	so there is less need to tabulate the results.

filehdf5/
	gX/
		gmodel_1/
			gX_mean # database: an fnlm array, with f.mean
			gX_sdev # database: an fnlm array, with f.sdev
		gmodel_2/
			gX_mean
			gX_sdev
		...
	fgs2/
		fSmodel_1/
			fgs_mean
			fgs_sdev
		fSmodel_2/
		...
	mcalI/
		mIndex_1/
			Ilvq_mean
			Ilvq_sdev
			mcalI_ell/
				ell_0/
					Iell_mean
					Iell_sdev
				ell_1/
				...
		mIndex_2/
			...
	mcalK/ #sort first by (gX, fgs2) model, then (mX, FDM_2)
		gXfs_(1,1)/
			mXFdm_(1,0)
				ell_0/
					mK_mean
					mK_sdev
				ell_1/
					...
			mXFdm_(1,2)
			...
		gXfs_(1,0)/
		...



