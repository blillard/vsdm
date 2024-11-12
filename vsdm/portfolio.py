"""Organizes an HDF5 dataset for an analysis combining gX and fs2 models.

"""

__all__ = ['Portfolio', 'str_to_bdict', 'str_to_dnames', 'update_namedict',
           'dname_manager', 'DEFAULT_TYPES', 'DNAME_F', 'DNAME_G', 'DNAME_I',
           'DNAME_W', 'TNAME_mcalI', 'TNAME_gX', 'TNAME_fs2', 'TNAME_generic',
           'TNAME_rotations', 'TNAME_mcalK', 'LM_IX_NAME', '_dset_exists']


import numpy as np
import h5py
import gvar

from .utilities import *

"""
    Global names for hdf5 groups and subgroups (types and databases)
    DNAME_: database name for saved data (e.g. <f|nlm>, gaussian Gnl, ...)
    LM_IX_NAME: maps the rows of DNAME_F to the corresponding (l, m).
"""
# dataset names for hdf5 files:
DNAME_F = 'fnlm'
DNAME_G = 'Gnl'
DNAME_I = 'Ilvq'
DNAME_W = 'wG'
LM_IX_NAME = 'lm_index'
# _dnames_ = ['DNAME_F', 'DNAME_G', 'DNAME_I', 'DNAME_W']
# saves gvar matrix to 'NAME_mean' and 'NAME_sdev'.
# subgroup 'type' names for hdf5 files:
TNAME_mcalI = 'mcalI'
TNAME_gX = 'gX'
TNAME_fs2 = 'fs2'
TNAME_generic = 'proj'
TNAME_rotations = 'rotations'
TNAME_mcalK = 'mcalK'
# _tnames_ = ['TNAME_mcalI', 'TNAME_gX', 'TNAME_fs2', 'TNAME_rotations', 'TNAME_mcalK']
DEFAULT_TYPES = [TNAME_mcalI, TNAME_gX, TNAME_fs2]

def str_to_bdict(strdict):
    """Converts hdf5.attrs dictionary to Basis.basis format."""
    # Here are all of the items that might be included in the hdf5 attrs:
    floats = ['u0', 'uMax', 't_eval']
    float_lists = ['uiList']
    ints = ['nMax', 'ellMax', 'neval', 'nitn_init', 'nitn']
    bools = ['zSymmetric', 'gaussianF']
    strings = ['type']
    out = {}
    for lbl in floats:
        if lbl in strdict:
            out[lbl] = float(strdict[lbl])
    for lbl in float_lists:
        if lbl in strdict:
            #convert string of list to list of strings
            flist = float(strdict[lbl]).replace('[', '').replace(']', '')
            list_str = flist.split(', ')
            out[lbl] = [float(str_item) for str_item in list_str]
    for lbl in ints:
        if lbl in strdict:
            out[lbl] = int(strdict[lbl])
    for lbl in bools:
        if lbl in strdict:
            out[lbl] = bool(strdict[lbl])
    for lbl in strings:
        if lbl in strdict:
            out[lbl] = strdict[lbl]
    return out

def str_to_dnames(strdict):
    """Converts hdf5 group.attrs dictionary to dname dictionary.

    This function converts str-valued 'strdict' to int-valued 'dnames'.

    Format:
        dset_names[dname]: initialized to 1 when a dataset with
            name 'dname' is added to the group, saved to group.attrs.
        When adding new datasets to 'group', check if 'dname'
            already exists in group.attrs. If so, then perform
                dset_names[dname] += 1
            and then save the dataset as:
                dname_new = dname + '_' + dset_names[dname].
        For completeness, could add 'dname_new' to group.attrs.
    Note: model 'dname' is string valued for both dicts.
    """
    dnames = {}
    for dname,howmanyofit in strdict.items():
        try:
            dn_index = int(howmanyofit)
        except ValueError:
            #skip anything that doesn't like being turned into an integer
            # in case group.attrs is used to hold unrelated attributes
            continue
        dnames[dname] = dn_index
    return dnames

def dname_manager(dname_dict, dname, sepchar='__', delete=False):
    """Reads dataset names from dname_dict, modifies dname or dname_dict.

    Format:
        dname: str-valued dictionary key
        dname_dict: int-valued dictionary of name multiplicities
            dname_dict[dname] = # of datasets with 'dname' as name base.
            initialized to 1 upon creation of 'dname'.
        sepchar: appended to 'dname' for dname_dict[dname] > 1 entries.
            default: '__'. Avoid anything already being used in the names.
    Used to label new files as 'dname_2', 'dname_3', etc.
        e.g. as opposed to 'dname_1_1', 'dname_1_1_1', ...

    When called using delete=True, it removes dname from dname_dict.
    """
    dict_out = dname_dict # avoid modifying the input
    if dname in dict_out:
        # dname can be deleted from the dictionary:
        if delete:
            del dict_out[dname]
            return dict_out, dname
        # Otherwise, dname needs an integer-valued suffix...
        name_in_dict = True
        while name_in_dict:
            #check if i += 1 is available :
            dict_out[dname] += 1
            dname_new = dname + sepchar + str(dict_out[dname])
            name_in_dict = (dname_new in dict_out)
            if not name_in_dict:
                # then this is a good name.
                dname_out = dname_new
            # otherwise, repeat the while loop and try i += 1.
        # ... and the new name is added to dname_dict
        dict_out[dname_out] = 1
    else:
        # then add it to the list
        dict_out[dname] = 1
        dname_out = dname # and keep the dataset name unchanged
    return dict_out, dname_out

def update_namedict(hdf5model, sepchar='__'):
    """Creates a 'namedict' for group 'model' if it does not exist.

    hdf5model: an hdf5 group at the 'model' level,
        i.e. one that contains datasets

    * Only needed for hdf5 files modified outside Portfolio framework.
    """
    name_dict = str_to_dnames(hdf5model.attrs) # contains any existing name_dict entries
    for key in hdf5model.keys(): # list of dbases or subgroups
        # assuming hdf5model[key] is a dataset, not a group.
        if key in name_dict:
            continue
        # else: add 'key' to name_dict
        name_dict[key] = 1
    # Next, check for duplications of name_dict[key]:
    for key in name_dict:
        i = name_dict[key]
        name_in_hdf5 = True
        while name_in_hdf5:
            i += 1 # look for next file name
            nextname = key + sepchar + str(i)
            if nextname in hdf5model.keys():
                # then name_dict entry is out of date!
                name_dict[key] = i
            else:
                name_in_hdf5 = False
                # stop, now name_dict is up to date
    """
    Note 1: this leaves name_dict ready to fill the first open name spot.
    It may appear in the middle of a sequence, _1, _2, _3, _6, _7, ...,
        if items '4' and '5' were deleted.
    dname_manager handles this: before trying to name something _6 it checks
        whether _6 already exists, and moves on to the first available integer.

    Note 2: This also adds subgroup names to the name_dict.
        This has little effect: dname_manager does not manage subgroups,
        so any attempt to name a dataset with a group name will lead to error.

    Note 3. Deleting a database from the hdf5 file does not remove its name
        from the dname_manager. The name entry also needs to be deleted from
        group.attrs.
    """
    return name_dict


def _dset_exists(hdf5file, tname, model, dname):
    """Looks for dataset 'dname' from hdf5, returns True if it exists."""
    with h5py.File(hdf5file,'r') as fhd5:
        mgroup = fhd5[tname + '/' + model]
        if dname in mgroup.keys():
            return True
        else:
            return False


class Portfolio():
    """Maintains a unified hdf5 file structure for a rate analysis.

    Intended for one choice of V and one choice of Q basis per hdf5 file.

    File structure hierarchy:
    0. 'hdf5file'. Intended that all datasets use the same (V, Q) basis.
    1. 'type': 'gX', 'fs2' or 'mcalI', and possibly 'rotations'
    2. 'model', for different values of the parameters.
            e.g. gX with different halo parameters, or stream inclusions;
            or, fs2 with different material form factors or excited states;
            or mcalI with different DM particle models & values of DeltaE
    3. datasets, usually _mean and _sdev for gvar-valued data.

    Additional levels of subgroups can be added within 'model': e.g.
        'simulation/v1', 'simulation/v2', 'simulation/for_real_this_time',
        or 'molecules/t-stilbene/state_s1', 'molecules/t-stilbene/state_s2'.
    I recommend restricting the 'type' list to the three basic objects,
        maybe adding a category 'rotations' for precalculated WignerG matrices
        (especially if installing quaternionic is difficult for people)
    * Extra 'types' can be specified to __init__().
    * Portfolio only uses the top-level groups 'gX', 'fs2' and 'mcalI',
        plus anything added to 'extra_types' dict.
        Other top-level groups will be ignored.

    example_hdf5file/
        gX/
            SHM/
                fnlm_mean, fnlm_sdev
            SHM_annual/
                Jan/, Feb/, Mar/, Apr/, May/, ...
                    fnlm_mean, fnlm_sdev
            streams/
                model1/, model2/, model3/ ...
                    fnlm_mean, fnlm_sdev
            simulation_A/
                fnlm_mean, fnlm_sdev
        fs2/
            particles_in_boxes/
                boxshape_1/, boxshape_2/, boxshape_3/...
                    fnlm_mean, fnlm_sdev
            trans-stilbene/
                s1/, s2/, ..., s7/, s8/
                    fnlm_mean, fnlm_sdev
        mcalI/
            (DeltaE_eV)/
                (mX_MeV, fdm_n)/
                    Ilvq_mean
                (mX_MeV, fdm_n)/
                    Ilvq_mean
        rotations/
            list_of_saved_R_values
            0/
                R, wG_l0, wG_l1, wG_l2, ..., wG_lMax
            1/
                R, wG_l0, wG_l1, wG_l2, ..., wG_lMax
        mcalK/
            (gX_model, fs2_model)/
                    (mX, fdm_n)/
                        mcalK_l0_mean, mcalK_l1_mean, mcalK_l2_mean, ...
                        mcalK_l0_sdev, mcalK_l1_sdev, mcalK_l2_sdev, ...

    Above, items in () indicate that the model name is a tuple, e.g.
        (DeltaE_eV) -> '1.0', '4.2', '6.414', ...
        (mX_MeV, fdm_n) -> (1.0, 0), (1.0, 2), (2.0, 0), ..., etc.
    If the WignerG matrices are to be provided for some list of R, with
        (R) -> (1, 0, 0, 0), (0.25, 0.25, -0.25, -0.25),
    here using unit quaternions to describe the rotation, one possible
        approach is shown above.
    Alternatively, each matrix wG_l can be turned into a 3d array, e.g.
        rotations/
            list_of_saved_R_values
            wG_R_l0, wG_R_l1, wG_R_l2, ..., wG_R_lMax
    The latter method is more compact, though less adaptable for inserting
        new rotations in the list after the fact.

    Usually mcalK is not provided. It is a proxy for the scattering rate,
        mu(R) ~ sum_l Tr(G_l * mcalK_l), where G_l(R) is the WignerG matrix
        for a given rotation 'R'.
    It is easy to calculate from gX, fs2, and mcalI, so it is less important
        to save the evaluated values. And, if there are many gX and fs2 models,
        a complete record of all possible mcalK matrices could take up a
        large amount of storage space.
    On the other hand, if NMAX is large and ELLMAX is not,
        mcalK may be more compact than the other items.

    Attributes:
        dnames_record: tracks the model and dataset names that have been used.
            includes names that have been created in add_folio
                or that have been imported
            Local parts of dnames_record are saved to the group.attrs
                for group = type/model, rather than type.attrs.
            On introduction of new dname, initialize
                dnames_record[type][model][dname] = 1
            Any attempts to create datasets of the same name will +=1 it
    """
    def __init__(self, hdf5file, extra_types=[], sepchar='__'):
        self.hdf5file = hdf5file
        self.sepchar = sepchar # default set to '__' for uniqueness
        self.types = DEFAULT_TYPES # list containing ['gX', 'fs2', ...]
        if extra_types is not None:
            # for example, to initialize with directory for mcalK
            for newtype in extra_types:
                if newtype not in self.types:
                    self.types += [newtype]
        #Make the highest level groups, if they don't already exist
        with h5py.File(hdf5file,'a') as hdf5: #create file if it does not exist
            for tname in self.types:
                if tname not in hdf5:
                    group = hdf5.create_group(tname)

    def add_folio(self, tname, model, dname, data=None, attrs={},
                  update_ndict=False):
        """Method for adding entries to the Portfolio.

        Arguments:
            tname, model: specifies the directory 'type/model/' for dataset
            dname, data: the intended dataset name and data
                if dname is already taken for this model, use dname_manager()
            attrs: to save to dbase.attrs dict.
            update_ndict: whether to run update_namedict() on the model.attrs

        File format:
            hdf5file/type/model/dbase
            group = hdf5file/type/model
            group.attrs: stores name_dict for all datasets in group
            dbase.attrs: stores parameter info for data

        'name_record':
            Any subgroup containing datasets should have a 'name_record' within
                the subgroup.attrs dict.
            'add_folio' creates or updates it whenever it adds a new dataset.
            Intermediate groups created with compound model names,
                e.g. 'benzene' in model = 'benzene/s1',
                only get a name_record if a dataset is added to it ('benzene')

        """
        dname_out = dname
        with h5py.File(self.hdf5file,'a') as hdf5:
            groupName = tname + '/' + model
            if groupName in hdf5:
                group = hdf5[groupName]
                if update_ndict:
                    name_record = update_namedict(group, sepchar=self.sepchar)
                else:
                    name_record = str_to_dnames(group.attrs)
            else:
                group = hdf5.create_group(groupName)
                name_record = {}
            name_record, dname = dname_manager(name_record, dname)
            group.attrs.update(name_record) # save the update to group.attrs
            # if dbase_name is already used, then dname_manager has renamed it.
            if data is not None:
                dset = group.create_dataset(dname, data=data)
            dset.attrs.update(attrs) # external info saved to new dset.attrs
            dname_out = dname # if dname_manager updated dname, want to know
        return dname_out

    def read_folio(self, tname, model, dname):
        """Reads dataset 'dname' from hdf5. Returns data and dname.attrs."""
        data_attr = {}
        with h5py.File(self.hdf5file,'r') as fhd5:
            mgroup = fhd5[tname + '/' + model]
            data = np.array([row for row in mgroup[dname]])
            for lbl,val in mgroup[dname].attrs.items():
                data_attr[lbl] = val #import the basis metadata
            return data, data_attr

    def delete_folio(self, tname, model, dname):
        """Deletes dataset 'dname' from hdf5."""
        with h5py.File(self.hdf5file,'a') as fhd5:
            mgroup = fhd5[tname + '/' + model]
            if dname in mgroup.keys():
                del mgroup[dname]
                name_record, dname = dname_manager(group.attrs, dname, delete=True)
                group.attrs.update(name_record) # save the update to group.attrs
                print("Deleted dataset '{}' from group '{}'.".format(dname,model))
            else:
                print("No dataset '{}' in group '{}'.".format(dname,model))

    def update_folio(self, tname, model, dname, newdata={}, attrs={}):
        """Modifies the dataset 'dname' with the provided newdata.

        Arguments:
            tname, model, dname: group name and dataset
            newdata: a dict object of form {index: value}, for
                dname[index] = value
                Any keys in newdata with the wrong shape for array index
                    will be ignored.
            attrs: any items to add to the dname.attrs dict.
        """
        data_attr = {}
        with h5py.File(self.hdf5file,'r+') as fhd5:
            mgroup = fhd5[tname + '/' + model]
            dbase = mgroup[dname] # the dataset
            dshape = np.shape(dbase) # initial size of dataset array
            dim = len(dshape) # dimensionality of array
            for index,value in newdata.items():
                # make sure index is valid
                if len(index)>dim:
                    print("Warning: data includes invalid entries.")
                    continue
                bad_index = False
                # a correct index length doesn't guarantee its suitability:
                for i in range(dim):
                    if type(index[i]) is not int:
                        bad_index = True
                if bad_index:
                    print("Warning: data includes invalid entries.")
                    continue
                # Check whether this index fits within dshape:
                newshape = compare_index_to_shape(index, dshape)
                if newshape != dshape:
                    dbase.resize(newshape)
                    dshape = newshape
                # Add the new value
                dbase[index] = value
            # At the end, add any attrs to the dbase.attrs dict
            dbase.attrs.update(attrs)


    def read_gvar(self, tname, modelName, d_pair):
        """Reads hdf5 datasets d_pair, returns gvar array.

        d_pair: pair of _mean and _sdev files in hdf5file/tname/modelName
            if len(dnames)==1, read d_pair as _mean, with .sdev = 0.

        Uses joinGVARarray to combine into one gvar array.
        """
        data_attr = {}
        f_mean, data_attr = self.read_folio(tname, modelName, d_pair[0])
        if len(d_pair)==1:
            fnlm = joinGVARarray(f_mean, None)
            return fnlm, data_attr
        # else:
        f_sdev, tobeignored = self.read_folio(tname, modelName, d_pair[1])
        fnlm = joinGVARarray(f_mean, f_sdev)
        #data_attr is str-valued.
        return fnlm, data_attr

    def update_gvar(self, tname, model, d_pair, newdata={}, attrs={}):
        """Modifies the datasets d_pair with the new gvar newdata.

        d_pair: pair of _mean and _sdev files in hdf5file/tname/modelName
            Needs to be a pair: use update_folio to edit only _mean

        Arguments:
            tname, model, d_pair: group name and dataset
            newdata: a dict object of form {index: value_gvar}
            attrs: any items to add to the dname.attrs dicts.
        """
        assert(len(d_pair)==2), "Error: d_pair needs 2 elements for 'update'."
        data_mean = {}
        data_sdev = {}
        for index,value in newdata.items():
            if type(value) is gvar._gvarcore.GVar:
                data_mean[index] = value.mean
                data_sdev[index] = value.sdev
            elif (type(value) == float) or (type(value) == int):
                data_mean[index] = value
                data_sdev[index] = 0
            else:
                # skip anything with incompatible 'value'
                continue
        self.update_folio(tname, model, d_pair[0],
                          newdata=data_mean, attrs=attrs)
        self.update_folio(tname, model, d_pair[1],
                          newdata=data_sdev, attrs=attrs)






#
