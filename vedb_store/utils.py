"""vedb_store utility functions"""
from . import options
from collections import OrderedDict
import pathlib
import numpy as np
import warnings
import yaml
import copy
import os


SESSION_FIELDS = [#'study_site', # add back?
                    #'scene',
                    'location',
                    'task',
                    ]
RECORDING_FIELDS = ['tilt_angle',
                    'lens',
                    'rig_version',
                    ]

METADATA_DEFAULTS = dict(tilt_angle='100',
						 lens='new',
						 )


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

def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays"""
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts"""
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out


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
                to_remove.append(v)
                new_dict[new_value] = dd[old_value]
        for x in to_remove:
            if x in dd:
                _ = dd.pop(x)
        new_dict.update(**dd)
        new_dictlist.append(new_dict)
    return new_dictlist


def dictlist_to_dataframe(dictlist, mapping=mapping_pupil_to_df):
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
	pass

def get_frame_indices(start_time, end_time, all_time):
	"""Finds start and end indices for frames that are between `start_time` and `end_time`
	
	Note that `end_frame` returned will be the first frame that occurs after
	end_time, such that some data[start_frame:end_frame] will span the range 
	between `start_time` and `end_time`. 

	Parameters
	----------
	start_time: scalar
		time after which to select frames
	end_time: scalar
		time before which to select frames
	all_time: array-like
		full array of timestamps for data into which to index.

	"""
	ti = (all_time > start_time) & (all_time < end_time)
	time_clipped = all_time[ti]
	indices, = np.nonzero(ti)
	start_frame, end_frame = indices[0], indices[-1] + 1
	return start_frame, end_frame 


def _check_dict_list(dict_list, n=1, **kwargs):
    tmp = dict_list
    for k, v in kwargs.items():
        tmp = [x for x in tmp if (hasattr(x, k)) and (getattr(x, k) == v)]
    if n is None:
        return tmp
    if len(tmp) == n:
        if n == 1:
            return tmp[0]
        else:
            return tmp
    else:
        raise ValueError('Requested number of items not found')

def load_pipeline_elements(session,
                           pupil_param_tag='plab_default',
                           pupil_drift_param_tag=None,
                           cal_marker_param_tag='circles_halfres',
                           cal_marker_filter_param_tag='cluster_default',
                           calib_param_tag='monocular_tps_default',
                           calibration_epoch=0,
                           val_marker_param_tag='checkerboard_halfres',
                           val_marker_filter_param_tag='basic_split',
                           mapping_param_tag='default_mapper',
                           error_param_tag='smooth_tps_default',
                           dbi=None,
                           is_verbose=True,
                           ):
    if dbi is None:
        dbi = session.dbi
    verbosity = copy.copy(dbi.is_verbose)
    dbi.is_verbose = is_verbose >= 1
    
    # Get all documents associated with session
    session_docs = dbi.query(session=session._id)
    # Create outputs dict
    outputs = dict(session=session)
    
    if pupil_param_tag is not None:
        outputs['pupil'] = {}
        for eye in ['left', 'right']:
            try:
                print("> Searching for %s pupil (%s)" % (eye, pupil_param_tag))
                outputs['pupil'][eye] = _check_dict_list(session_docs, 
                                                         n=1,
                                                         type='PupilDetection', 
                                                         tag=pupil_param_tag, 
                                                         eye=eye)
                print(">> FOUND %s pupil" % (eye))
            except:
                print('>> NOT found')

    if cal_marker_param_tag is not None:
        try:
            print("> Searching for calibration markers...")
            outputs['calibration_marker_all'] = _check_dict_list(
                session_docs, n=1, tag=cal_marker_param_tag, epoch='all')
            print(">> FOUND it")
        except:
            print('>> NOT found')

    if cal_marker_filter_param_tag is not None:
        try:
            print("> Searching for filtered calibration markers...")
            cfiltered_tag = '-'.join([cal_marker_param_tag,
                                      cal_marker_filter_param_tag])
            outputs['calibration_marker_filtered'] = _check_dict_list(
                session_docs, n=1, tag=cfiltered_tag, epoch=calibration_epoch)
            print(">> FOUND it")
        except:
            print('>> NOT found')

    if val_marker_param_tag is not None:
        try:
            if isinstance(val_marker_param_tag, tuple):
                for t in val_marker_param_tag:
                    print("> Searching for validation markers...")
                    tmp = _check_dict_list(
                        session_docs, n=1, tag=t, epoch='all')
                    outputs['validation_marker_all'] = tmp
                    if not tmp.failed:
                        break
            else:
                print("> Searching for validation markers...")
                outputs['validation_marker_all'] = _check_dict_list(
                    session_docs, n=1, tag=val_marker_param_tag, epoch='all')

            print(">> FOUND it")
        except:
            print('>> NOT found')

    if val_marker_filter_param_tag is not None:
        try:
            print("> Searching for filtered validation markers...")
            vfiltered_tag = '-'.join([val_marker_param_tag,
                                    val_marker_filter_param_tag])
            tmp = _check_dict_list(session_docs, n=None, tag=vfiltered_tag)
            if len(tmp) == 0:
                1/0  # error out, nothing found
            tmp = sorted(tmp, key=lambda x: x.epoch)
            outputs['validation_marker_filtered'] = tmp
            print(">> FOUND %d" % (len(tmp)))
        except:
            print(">> NOT found")

    if calib_param_tag is not None:
        if 'monocular' in calib_param_tag:
            eyes = ['left', 'right']
        else:
            eyes = ['both']

        for ie, eye in enumerate(eyes):
            if ie == 0:
                outputs['calibration'] = {}
                outputs['gaze'] = {}
                outputs['error'] = {}
            try:
                print("> Searching for %s calibration" % eye)
                calib_tag_full = '-'.join([pupil_param_tag, 
                                           cal_marker_param_tag, 
                                           cal_marker_filter_param_tag,
                                           calib_param_tag])
                outputs['calibration'][eye] = _check_dict_list(session_docs, 
                    n=1,
                    type='Calibration',
                    tag=calib_tag_full,
                    eye=eye,
                    epoch=calibration_epoch)
                print(">> FOUND %s calibration" % eye)
            except:
                print('>> NOT found')
            try:
                print("> Searching for %s gaze" % eye)
                gaze_tag_full = '-'.join([pupil_param_tag,
                                          cal_marker_param_tag,
                                          cal_marker_filter_param_tag,
                                          calib_param_tag,
                                          mapping_param_tag,
                                          ])
                outputs['gaze'][eye] = _check_dict_list(session_docs, n=1, 
                    type='Gaze', tag=gaze_tag_full, eye=eye)
                print(">> FOUND %s gaze" % eye)
            except:
                print('>> NOT found')

            try:
                print("> Searching for error")
                err_tags = [pupil_param_tag, 
                            cal_marker_param_tag, 
                            cal_marker_filter_param_tag,
                            calib_param_tag, 
                            mapping_param_tag, 
                            val_marker_param_tag, 
                            val_marker_filter_param_tag, 
                            error_param_tag]
                # Skip any steps not provided? Likely to cause bugs below
                err_tag = '-'.join(err_tags)
                tmp = _check_dict_list(session_docs, n=None,
                                        tag=err_tag, eye=eye)
                if len(tmp) == 0:
                    1/0  # error out, nothing found                                        
                err = sorted(tmp, key=lambda x: x.epoch)
                outputs['error'][eye] = err
                print(">> FOUND it")
            except:
                print(">> NO error found for %s"%eye)

    for field in ['pupil', 'calibration', 'gaze']:
        if (field in outputs) and len(outputs[field]) == 0:
            _ = outputs.pop(field)

    if 'error' in outputs:
        if 'left' in outputs['error']:
            n_err_left = len(outputs['error']['left'])
        else:
            n_err_left = 0
        if 'right' in outputs['error']:
            n_err_right = len(outputs['error']['right'])
        else:
            n_err_right = 0
        if (n_err_left != n_err_right):
            print("Error mismatch: %d on left, %d on right" %
              (n_err_left, n_err_right))
            _ = outputs.pop("error")

    dbi.is_verbose = verbosity

    return outputs

def get_time_split(total_time, n_parts, frac_to_vary=0.05):
    """
    stddev, if not None, overrules frac_to_vary
    """
    mean_duration = total_time / n_parts
    stddev = mean_duration * frac_to_vary
    vec = np.random.normal(loc=mean_duration, scale=stddev, size=(n_parts))
    vec *= (total_time / vec.sum())
    return vec