#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#@author: kushal

#Chatzigeorgiou Group
#Sars International Centre for Marine Molecular Biology

#GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

"""
A bunch of functions from various Mesmerize modules that are used 
in the notebooks that the entire Mesmerize package doesn't need to 
be installed to run these on binder etc.
"""


import os
from typing import *
import numpy as np
import h5py
import json
import pandas as pd
from warnings import warn
import traceback
from sklearn.metrics import pairwise_distances
from collections import OrderedDict
from matplotlib import cm as matplotlib_color_map
from itertools import product

qual_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
              'tab20c']

class HdfTools:
    """Functions for saving and loading HDF5 data"""
    @staticmethod
    def save_dataframe(path: str, dataframe: pd.DataFrame, metadata: Optional[dict] = None,
                       metadata_method: str = 'json', raise_meta_fail: bool = True):
        """
        Save DataFrame to hdf5 file along with a meta data dict.

        Meta data dict can either be serialized with json and stored as a str in the hdf5 file, or recursively saved
        into hdf5 groups if the dict contains types that hdf5 can deal with. Experiment with both methods and see what works best

        Currently the hdf5 method can work with these types: [str, bytes, int, float, np.int, np.int8, np.int16,
        np.int32, np.int64, np.float, np.float16, np.float32, np.float64, np.float128, np.complex].

        If it encounters an object that is not of these types it will store whatever that object's __str__() method
        returns if on_meta_fail is False, else it will raise an exception.

        :param path:            path to save the file to
        :param dataframe:       DataFrame to save in the hdf5 file
        :param metadata:        Any associated meta data to store along with the DataFrame in the hdf5 file
        :param metadata_method: method for storing the metadata dict, either 'json' or 'recursive'
        :param raise_meta_fail: raise an exception if recursive metadata saving encounters an unsupported object
        """
        if os.path.isfile(path):
            raise FileExistsError

        f = h5py.File(path, mode='w')

        f.create_group('DATAFRAME')

        if metadata is not None:
            mg = f.create_group('META')
            mg.attrs['method'] = metadata_method

            if metadata_method == 'json':
                bad_keys = []
                for k in metadata.keys():
                    try:
                        mg.create_dataset(k, data=json.dumps(metadata[k]))
                    except TypeError as e:
                        bad_keys.append(str(e))

                if len(bad_keys) > 0:
                    bad_keys = '\n'.join(bad_keys)
                    raise TypeError(f"The following meta data keys are not JSON serializable\n{bad_keys}")


            elif metadata_method == 'recursive':
                HdfTools._dicts_to_group(h5file=f, path='META/', d=metadata, raise_meta_fail=raise_meta_fail)

        f.close()

        dataframe.to_hdf(path, key='DATAFRAME', mode='r+')

    @staticmethod
    def load_dataframe(filepath: str) -> Tuple[pd.DataFrame, Union[dict, None]]:
        with h5py.File(filepath, 'r') as f:
            if 'META' in f.keys():

                if f['META'].attrs['method'] == 'json':
                    ks = f['META'].keys()
                    metadata = dict.fromkeys(ks)
                    for k in ks:
                        metadata[k] = json.loads(f['META'][k][()])

                elif f['META'].attrs['method'] == 'recursive':
                    metadata = HdfTools._dicts_from_group(f, 'META/')

            else:
                metadata = None
        df = pd.read_hdf(filepath, key='DATAFRAME', mode='r')

        return (df, metadata)

    @staticmethod
    def save_dict(d: dict, filename: str, group: str, raise_type_fail=True):
        """
        Recursively save a dict to an hdf5 group.

        :param d:        dict to save
        :param filename: filename
        :param group:    group name to save the dict to
        :param raise_type_fail: whether to raise if saving a piece of data fails
        """
        if os.path.isfile(filename):
            raise FileExistsError

        with h5py.File(filename, 'w') as h5file:
            HdfTools._dicts_to_group(h5file, f'{group}/', d, raise_meta_fail=raise_type_fail)

    @staticmethod
    def _dicts_to_group(h5file: h5py.File, path: str, d: dict, raise_meta_fail: bool):
        for key, item in d.items():

            if isinstance(item, np.ndarray):

                if item.dtype == np.dtype('O'):
                    # see if h5py is ok with it
                    try:
                        h5file[path + key] = item
                        # h5file[path + key].attrs['dtype'] = item.dtype.str
                    except:
                        msg = f"numpy dtype 'O' for item: {item} not supported by HDF5\n{traceback.format_exc()}"

                        if raise_meta_fail:
                            raise TypeError(msg)
                        else:
                            h5file[path + key] = str(item)
                            warn(f"{msg}, storing whatever str(obj) returns.")

                # numpy array of unicode strings
                elif item.dtype.str.startswith('<U'):
                    h5file[path + key] = item.astype(h5py.special_dtype(vlen=str))
                    h5file[path + key].attrs['dtype'] = item.dtype.str  # h5py doesn't restore the right dtype for str types

                # other types
                else:
                    h5file[path + key] = item
                    # h5file[path + key].attrs['dtype'] = item.dtype.str

            # single pieces of data
            elif isinstance(item, (str, bytes, int, float, np.int, np.int8, np.int16, np.int32, np.int64, np.float,
                                   np.float16, np.float32, np.float64, np.float128, np.complex)):
                h5file[path + key] = item

            elif isinstance(item, dict):
                HdfTools._dicts_to_group(h5file, path + key + '/', item, raise_meta_fail)

            # last resort, try to convert this object to a dict and save its attributes
            elif hasattr(item, '__dict__'):
                HdfTools._dicts_to_group(h5file, path + key + '/', item.__dict__, raise_meta_fail)

            else:
                msg = f"{type(item)} for item: {item} not supported not supported by HDF5"

                if raise_meta_fail:
                    raise ValueError(msg)

                else:
                    h5file[path+key] = str(item)
                    warn(f"{msg}, storing whatever str(obj) returns.")

    @staticmethod
    def load_dict(filename: str, group: str) -> dict:
        """
        Recursively load a dict from an hdf5 group.

        :param filename: filename
        :param group:    group name of the dict
        :return:         dict recursively loaded from the hdf5 group
        """
        with h5py.File(filename, 'r') as h5file:
            return HdfTools._dicts_from_group(h5file, f'{group}/')

    @staticmethod
    def _dicts_from_group(h5file: h5py.File, path: str) -> dict:
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                if item.attrs.__contains__('dtype'):
                    ans[key] = item[()].astype(item.attrs['dtype'])
                else:
                    ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = HdfTools._dicts_from_group(h5file, path + key + '/')
        return ans


def pad_arrays(a: np.ndarray, method: str = 'random', output_size: int = None, mode: str = 'minimum',
               constant: Any = None) -> np.ndarray:
    """
    Pad all the input arrays so that are of the same length. The length is determined by the largest input array.
    The padding value for each input array is the minimum value in that array.

    Padding for each input array is either done after the array's last index to fill up to the length of the
    largest input array (method 'fill-size') or the padding is randomly flanked to the input array (method 'random')
    for easier visualization.

    :param a: 1D array where each element is a 1D array
    :type a: np.ndarray

    :param method: one of 'fill-size' or 'random', see docstring for details
    :type method: str

    :param output_size: not used

    :param mode: one of either 'constant' or 'minimum'.
                 If 'minimum' the min value of the array is used as the padding value.
                 If 'constant' the values passed to the "constant" argument is used as the padding value.
    :type mode: str

    :param constant: padding value if 'mode' is set to 'constant'
    :type constant: Any

    :return: Arrays padded according to the chosen method. 2D array of shape [n_arrays, size of largest input array]
    :rtype: np.ndarray
    """

    l = 0  # size of largest time series

    # Get size of largest time series
    for c in a:
        s = c.size
        if s > l:
            l = s

    if (output_size is not None) and (output_size < l):
        raise ValueError('Output size must be equal to larger than the size of the largest input array')

    # pre-allocate output array
    p = np.zeros(shape=(a.size, l), dtype=a[0].dtype)

    # pad each 1D time series
    for i in range(p.shape[0]):
        s = a[i].size

        if s == l:
            p[i, :] = a[i]
            continue

        max_pad_en_ix = l - s

        if method == 'random':
            pre = np.random.randint(0, max_pad_en_ix)
        elif method == 'fill-size':
            pre = 0
        else:
            raise ValueError('Must specific method as either "random" or "fill-size"')

        post = l - (pre + s)

        if mode == 'constant':
            p[i, :] = np.pad(a[i], (pre, post), 'constant', constant_values=constant)
        else:
            p[i, :] = np.pad(a[i], (pre, post), 'minimum')

    return p


def get_proportions(xs: Union[pd.Series, np.ndarray, list], ys: Union[pd.Series, np.ndarray, pd.Series],
                    xs_name: str = 'xs', ys_name: str = 'ys',
                    swap: bool = False, percentages: bool = True) -> pd.DataFrame:
    """
    Get the proportions of xs vs ys. xs & ys are categorical data.

    :param xs: data plotted on the x axis
    :type xs: Union[pd.Series, np.ndarray]

    :param ys: proportions of unique elements in ys are calculated per xs
    :type ys: Union[pd.Series, np.ndarray]

    :param xs_name: name for the xs data, useful for labeling the axis in plots
    :type xs_name: str

    :param ys_name: name for the ys data, useful for labeling the axis in plots
    :type ys_name: str

    :param swap: swap x and y
    :type swap: bool

    :return:   DataFrame that can be plotted in a proportions bar graph
    :rtype: pd.DataFrame
    """

    if len(xs) != len(ys):
        raise ValueError('Length of xs and ys must match exactly')

    if isinstance(xs, np.ndarray):
        if xs.ndim > 1:
            raise ValueError('Can only accept 1D numpy array')

    if isinstance(ys, np.ndarray):
        if ys.ndim > 1:
            raise ValueError('Can only accept 1D numpy array')

    if swap:
        xs, ys = ys, xs
        xs_name, ys_name = ys_name, xs_name

    df = pd.DataFrame({xs_name: xs, ys_name: ys})
    if percentages:
        props_df = df.groupby([xs_name, ys_name]).agg({ys_name: 'count'}).groupby(by=xs_name).apply(lambda x: (x / x.sum()) * 100).unstack()
        props_df.columns = props_df.columns.get_level_values(-1)
    else:
        props_df = df.groupby([xs_name, ys_name]).agg({ys_name: 'count'}).unstack()
        props_df.columns = props_df.columns.get_level_values(-1)

    return props_df


def auto_colormap(n_colors: int, cmap: str = 'hsv', spacing: str = 'uniform', alpha: float = 1.0) -> List[np.ndarray]:
    """
    If non-qualitative map: returns list of colors evenly spread through the chosen colormap.
    If qualitative map: returns subsequent colors from the chosen colormap

    :param n_colors: Numbers of colors to return
    :param cmap:     name of colormap

    :param spacing:  option: 'uniform' returns evenly spaced colors across the entire cmap range
                     option: 'subsequent' returns subsequent colors from the cmap
    :param alpha:    alpha level, 0.0 - 1.0

    :return:         List of colors as numpy array. Length of list equals n_colors
    """

    valid = ['uniform', 'subsequent']
    if spacing not in valid:
        raise ValueError(f'spacing must be one of either {valid}')

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be within 0.0 and 1.0')

    cm = matplotlib_color_map.get_cmap(cmap)
    cm._init()

    lut = (cm._lut).view(np.ndarray)

    lut[:, 3] *= alpha

    if spacing == 'uniform':
        if not cmap in qual_cmaps:
            cm_ixs = np.linspace(0, 210, n_colors, dtype=int)
        else:
            if n_colors > len(lut):
                raise ValueError('Too many colors requested for the chosen cmap')
            cm_ixs = np.arange(0, len(lut), dtype=int)
    else:
        cm_ixs = range(n_colors)

    colors = []
    for ix in range(n_colors):
        c = lut[cm_ixs[ix]]
        colors.append(c)

    return colors


def get_colormap(labels: iter, cmap: str, **kwargs) -> OrderedDict:
    """
    Get a dict for mapping labels onto colors

    Any kwargs are passed to auto_colormap()

    :param labels:  labels for creating a colormap. Order is maintained if it is a list of unique elements.
    :param cmap:    name of colormap

    :return:        dict of labels as keys and colors as values
    """
    if not len(set(labels)) == len(labels):
        labels = list(set(labels))
    else:
        labels = list(labels)

    colors = auto_colormap(len(labels), cmap, **kwargs)

    return OrderedDict(zip(labels, colors))


def get_sampling_rate(transmission, tolerance: Optional[float] = 0.1) -> float:
    """
    Returns the mean sampling rate of all data in a Transmission if it is within the specified tolerance. Otherwise throws an exception.

    :param transmission:    Transmission object of the data from which sampling rate is obtained.

    :param tolerance:       Maximum tolerance (in Hertz) of sampling rate variation between different samples
    :type tolerance:        float

    :return:                The mean sampling rate of all data in the Transmission
    :rtype:                 float
    """
    sampling_rates = []
    for db in transmission.history_trace.data_blocks:
        if transmission.history_trace.check_operation_exists(db, 'resample'):
            sampling_rates.append(transmission.history_trace.get_operation_params(db, 'resample')['output_rate'])
        else:
            r = pd.DataFrame(transmission.get_data_block_dataframe(db).meta.to_list())['fps'].unique()
            # if rates.size > 1:
            #     raise ValueError("Sampling rates for the data do not match")
            # else:
            sampling_rates.append(r)

    rates = np.hstack([sampling_rates])

    if np.ptp(rates) > tolerance:
        raise ValueError("Sampling rates of the data differ by "
                         "greater than the set tolerance of " + str(tolerance) + " Hz")

    framerate = float(np.mean(sampling_rates))

    return framerate


def get_frequency_linspace(transmission) -> Tuple[np.ndarray, float]:
    """
    Get the frequency linspace.
    Throwns an exception if all datablocks do not have the same linspace & Nyquist frequencies
    :param transmission: Transmission containing data from which to get frequency linspace
    :return: tuple: (frequency linspace as a 1D numpy array, nyquist frequency)
    :rtype:  Tuple[np.ndarray, float]
    """

    # Check that all sampling rates are equal
    get_sampling_rate(transmission)

    fs = []
    nqs = []

    for db in transmission.history_trace.data_blocks:
        params = transmission.history_trace.get_operation_params(db, 'rfft')

        f = params['frequencies']
        fs.append(np.array(f))

        nqs.append(params['nyquist_frequency'])

    if len(set(nqs)) > 1:
        raise ValueError("Nyquist frequency of all data blocks must match exactly")

    # Check that the discrete frequencies of all datablocks match exactly
    for i, j in product(*(range(len(fs)), )*2):
        if i == j:
            continue
        if not np.array_equal(fs[i], fs[j]):
            raise ValueError("Discrete frequencies of all data blocks must match exactly")

    return fs[0], nqs[0]

