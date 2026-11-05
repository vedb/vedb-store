"""vedb_store utility functions"""
from . import options
from collections import OrderedDict
import pathlib
import numpy as np
import warnings
import yaml
import copy
import os

# Utilities shared with vedb_gaze: imported (not copied) so both packages
# behave the same; `vedb_store.utils.<name>` keeps working
from vedb_gaze.utils import (
    dictlist_to_arraydict,
    arraydict_to_dictlist,
    match_time_points,
    get_frame_indices,
    onoff_from_binary,
    onoff_to_binary,
    time_to_index,
    _check_dict_list,
)


def read_yaml(fname):
    """Thin wrapper to safely read a yaml file into a dictionary"""
    out = dict()
    with open(fname,"r") as fid:
        out = yaml.safe_load(fid)
    return out

def specify_marker_epochs(folder, fps=30, write_to_folder=True):
	ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'too many']
	timestamps = np.load(os.path.join(folder, 'world_timestamps.npy'))
	marker_type = ['calibration', 'validation']
	markers = {}
	for mk in marker_type:
		markers[mk + '_orig_times'] = []
		for count in ordinals:
			print("\n=== %s %s epoch ==="%(count.capitalize(), mk))
			minsec_str = input('Please enter start of epoch as `min,sec` : ')
			min_start, sec_start = [float(x) for x in minsec_str.split(',')]
			minsec_str = input('Please enter end of epoch as `min,sec` : ')
			min_end, sec_end = [float(x) for x in minsec_str.split(',')]
			markers[mk + '_orig_times'].append([min_start *
			                                   60 + sec_start, min_end * 60 + sec_end])
			quit = input('Enter additional %s? (y/n): '%mk)
			if quit[0].lower() == 'n':
				break
		mka = np.array(markers[mk + '_orig_times']).astype(int)
		markers[mk + '_frames'] = [[int(a), int(b)] for (a, b) in mka * fps]
		markers[mk + '_times'] = [[int(np.floor(a)), int(np.ceil(b))] for (a, b) in timestamps[mka * fps]]
	if write_to_folder:
		yaml_file = pathlib.Path(folder) / 'marker_times.yaml'
		if yaml_file.exists():
			raise ValueError('File %s already exists! Please rename or remove it if you wish to overwrite.'%(str(yaml_file)))
		else:
			with open(yaml_file, mode='w') as fid:
				yaml.dump(markers, fid)
	return markers


mapping_pupil_to_df = {'eye_id': 'id',
                       'norm_pos_y': ('norm_pos', 1),
                       'norm_pos_x': ('norm_pos', 0),
                       'location_x': ('location', 0),
                       'location_y': ('location', 1),
                       'ellipse_center_x': ('ellipse', 'center', 0),
                       'ellipse_center_y': ('ellipse', 'center', 1),
                       'ellipse_axis_a': ('ellipse', 'axes', 0),
                       'ellipse_axis_b': ('ellipse', 'axes', 1),
                       'ellipse_angle': ('ellipse', 'angle'),
                       }

mapping_marker_to_df = {'eye_id': 'id',
                       'norm_pos_y': ('norm_pos', 1),
                       'norm_pos_x': ('norm_pos', 0),
                       'location_x': ('location', 0),
                       'location_y': ('location', 1),
                       'ellipse_center_x': ('ellipse', 'center', 0),
                       'ellipse_center_y': ('ellipse', 'center', 1),
                       'ellipse_axis_a': ('ellipse', 'axes', 0),
                       'ellipse_axis_b': ('ellipse', 'axes', 1),
                       'ellipse_angle': ('ellipse', 'angle'),
                       }

mapping_df_to_pupil = {'id': 'eye_id',
                       'norm_pos': ('norm_pos_x', 'norm_pos_y'),
                       'location': ('location_x', 'location_y'),
                       'ellipse': {'axes': ('ellipse_axis_a', 'ellipe_axis_b'),
                                   'angle': 'ellipse_angle',
                                   'center': ('ellipse_center_x', 'ellipse_center_y'),
                                   }
                       }

mapping_marker_to_df = {'norm_pos_y': ('norm_pos', 1),
                        'norm_pos_x': ('norm_pos', 0),
                        'location_x': ('location', 0),
                        'location_y': ('location', 1),
                        'ellipse_center_x': ('ellipse', 'center', 0),
                        'ellipse_center_y': ('ellipse', 'center', 1),
                        'ellipse_axis_a': ('ellipse', 'axes', 0),
                        'ellipse_axis_b': ('ellipse', 'axes', 1),
                        'ellipse_angle': ('ellipse', 'angle'),
                        }


def dataframe_to_dictlist(dataframe, mapping=mapping_df_to_pupil):
    """Convert a dataframe to a list of dictionaries"""
    dictlist = []
    for index, row in dataframe.iterrows():
        tmp = {}
        for new_value, old_value in mapping.items():
            if isinstance(old_value, tuple):
                tmp[new_value] = [row[v] for v in old_value]
            elif isinstance(old_value, dict):
                sub_dict = {}
                for nv, ov in old_value.items():
                    if isinstance(ov, tuple):
                        sub_dict[nv] = [row[v] for v in ov]
                    else:
                        sub_dict[nv] = row[ov]
                tmp[new_value] = sub_dict
            elif isinstance(old_value, str):
                tmp[old_value] = row[new_value]
        dictlist.append(tmp)
    return dictlist


def dataframe_to_arraydict(dataframe, mapping=mapping_df_to_pupil):
    """Convert from dataframe to dict of arrays"""
    dictlist = dataframe_to_dictlist(dataframe, mapping=mapping)
    arraydict = dictlist_to_arraydict(dictlist)
    return arraydict


def arraydict_to_dataframe(arraydict, mapping=mapping_pupil_to_df):
    """Convert from dict of arrays to a dataframe
    
    Parameters
    ----------
    """
    import pandas as pd
    if mapping is not None:
        dictlist = arraydict_to_dictlist(arraydict)
        new_dictlist = remap_dict_values(dictlist, mapping=mapping)
        new_arraydict = dictlist_to_arraydict(new_dictlist)
    else:
        new_arraydict = arraydict
    df = pd.DataFrame(data=new_arraydict)
    return df


def remap_dict_values(dictlist, mapping=mapping_pupil_to_df):
    """Change values stored in each dictionary in a list of dicts
    
    For example, map a 2-tuple for 'location' to 'location_x' and 'location_y' keys
    in a new dictionary

    Parameters
    ----------
    dictlist : list
        list of dictionaries to have keys & values remapped
    mapping: dict
        dictionary governing mapping keys and values from one dict to anther
    
    """
    if mapping is None:
        mapping = {}
    new_dictlist = []
    for dd in dictlist:
        new_dict = {}
        to_remove = []
        for new_value, old_value in mapping.items():
            if isinstance(old_value, tuple):
                n = len(old_value)
                if n == 2:
                    a, b = old_value
                    to_remove.append(a)
                    new_dict[new_value] = dd[a][b]
                elif n == 3:
                    a, b, c = old_value
                    to_remove.append(a)
                    new_dict[new_value] = dd[a][b][c]
            else:
                to_remove.append(old_value)
                new_dict[new_value] = dd[old_value]
        for x in to_remove:
            if x in dd:
                _ = dd.pop(x)
        new_dict.update(**dd)
        new_dictlist.append(new_dict)
    return new_dictlist


def dictlist_to_dataframe(dictlist, mapping=mapping_pupil_to_df):
    """Convert from list of dicts (e.g. pupil format) to a dataframe"""
    import pandas as pd
    if mapping is not None:
        # remap_dict_values pops keys, so work on copies of the input dicts
        new_dictlist = remap_dict_values([dict(d) for d in dictlist], mapping=mapping)
    else:
        new_dictlist = dictlist
    new_arraydict = dictlist_to_arraydict(new_dictlist)
    df = pd.DataFrame(data=new_arraydict)
    return df


def _is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def get_function(function_name):
    """Load a function to a variable by name

    Parameters
    ----------
    function_name : str
        string name for function (including module)
    """
    import importlib
    fn_path = function_name.split('.')
    module_name = '.'.join(fn_path[:-1])
    fn_name = fn_path[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, fn_name)
    return func


def get_nearest_index(timestamp, all_time):
    j = np.argmin(np.abs(all_time - timestamp))
    return j


def get_time_split(total_time, n_parts, frac_to_vary=0.05):
    """
    stddev, if not None, overrules frac_to_vary
    """
    mean_duration = total_time / n_parts
    stddev = mean_duration * frac_to_vary
    vec = np.random.normal(loc=mean_duration, scale=stddev, size=(n_parts))
    vec *= (total_time / vec.sum())
    return vec


