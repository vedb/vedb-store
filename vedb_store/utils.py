"""vedb_store utility functions"""
from . import options
from collections import OrderedDict
import pathlib
import numpy as np
import warnings
import yaml
import copy
import os


SESSION_FIELDS = ['study_site',
					'experimenter_id',
					'lighting',
					'scene',
					'instruction',
					]
SUBJECT_FIELDS = ['subject_id',
					'birth_year',
					'gender',
					'ethnicity',
					'IPD',
					'height',
					]
RECORDING_FIELDS = ['tilt_angle',
					'lens',
					'rig_version',
					]

METADATA_DEFAULTS = dict(tilt_angle='100',
						 lens='new',
						 )
METADATA_DEFAULTS.update(dict((field, 'None') for field in SUBJECT_FIELDS))

def parse_vedb_metadata(yaml_file, participant_file, raise_error=True, overwrite_user_info=False, default_values=None):
	"""Extract metadata data from yaml file, filling in missing fields

	Optionally backs up original file and creates a new file with filled-in fields

	Only works for metadata (regarding experiment, subject, etc), not recording
	devices, so far. 

	Parameters
	----------
	yaml_file : string
		path to yaml file
	raise_error : bool
		Whether to raise an error if fields are missing. True simply raises an error, 
		False allows manual input of missing fields. 
	overwrite_user_info : bool
		Whether to create a new user info file. Old user file will be saved as `user_info_orig.csv`
		unless that file already exists (in which case no backup will be created - only
		original is backed up)

	"""
	# Assure yaml_file is a pathlib.Path
	yaml_file = pathlib.Path(yaml_file)
	participant_file = pathlib.Path(participant_file)
	if default_values is None:
		default_values = METADATA_DEFAULTS
	with open(yaml_file, mode='r') as fid:
		yaml_doc = yaml.safe_load(fid)
	required_fields = SESSION_FIELDS + SUBJECT_FIELDS + RECORDING_FIELDS
	allowable_missing = ['IPD', 'rig_version', 'instruction', 'birth_year', 'lens', 'scene', 'ethnicity','height', 'gender'] # Temporarily...
	if 'metadata' in yaml_doc:
		# Current version, good.
		metadata = yaml_doc['metadata']
		metadata_location = 'base'
	elif 'metadata' in yaml_doc['commands']['record']:
		# Legacy config compatibility
		metadata = yaml_doc['commands']['record']['metadata']
		metadata_location = 'commands/record'
	else: 
		metadata = None
	if metadata is None:
		if raise_error:
			raise ValueError("Missing metadata: no yaml file in folder.")
		else:
			metadata = {}
	user_info = parse_user_info(participant_file)
	metadata.update(user_info)
	metadata_keys = list(metadata.keys())
	metadata_keys = [k for k in metadata_keys if (metadata[k] is not None) and metadata[k] != '']
	missing_fields = list(set(required_fields) - set(metadata_keys))
	if len(missing_fields) > 0:
		if raise_error: 
			# TEMP allow flexibility with some fields, we will crack down on this later
			for mf in allowable_missing:
				if mf in missing_fields:
					if mf in default_values:
						metadata[mf] = default_values[mf]
					else:
						metadata[mf] = 'unknown'
					warnings.warn(f"Missing '{mf}' for subject; consider collecting!")
					_ = missing_fields.pop(missing_fields.index(mf))
			if len(missing_fields) > 0:
				raise ValueError('Missing fields in subject meta-data: {}'.format(missing_fields))
		else:
			# Fill in missing fields manually
			print('Missing fields: ', missing_fields)
			for mf in missing_fields:
				if mf in default_values:
					default = default_values[mf]
				else:
					default = 'unknown'
				value = input("Enter value for %s [press enter for '%s', type 'abort' to quit]"%(mf, default)) or default
				if value.lower() in ('abort' "'abort'"):
					raise Exception('Manual quit.')
				metadata[mf] = value
		# Assure values are of proper types
		for key in metadata.keys():
			value = metadata[key]
			if isinstance(value, str) and value.lower() in ('none'):
				value = None
			if key in ('tilt_angle',):
				value = int(value)
			elif key in ('age', 'height', 'birth_year'):
				if value is not None and value not in ('unknown', 'None'):
					value = int(value)
			elif key in ('rig_version'):
				if value is not None and value not in ('unknown', 'None'):
					value = float(value)
			metadata[key] = value
		if overwrite_user_info:
			# Optionally replace user info file with new one
			new_participant_file = participant_file.parent / 'user_info_orig.csv'
			if new_participant_file.exists():
				# Get rid of current one, original is already saved
				participant_file.unlink()			
			else:
				# Create backup
				print('copying to %s'%new_participant_file)
				participant_file.rename(new_participant_file)			
			write_user_info(participant_file, metadata)
	return metadata


def parse_user_info(fname):
	"""Parse user_info csv file for session"""
	if fname is None:
		# Is it a good idea to just return an empty dict? Should we throw an error?
		return {}
	with open(fname) as fid:
		lines = fid.readlines()
		out = dict(tuple([y.strip().strip("'").strip('"') for y in x.split(',')]) for x in lines)
		for k in out.keys():
			if k in ['IPD']:
				try:
					out[k] = float(out[k])
				except ValueError:
					out[k] = None
			elif k in ['height', 'age', 'birth_year']:
				try:
					out[k] = int(out[k])
				except ValueError:
					out[k] = None
			elif k in ['experimenter_id','subject_id']:
				out[k] = out[k].upper()
				# Oneoff bullshit
				if out[k] == 'AFULLER':
					out[k] = 'AF'
				elif out[k] == 'NUDNOUIL':
					out[k] = 'IN'
			elif k in ['ethnicity', 'gender']:
				out[k] = out[k].lower()
			if k == 'gender':
				if out[k] in ('f',):
					out[k] = 'female'
				elif out[k] in ('m',):
					out[k] = 'male'
	_ = out.pop('key')
	return out



# def standardize_metadata_fields(user_info):
# 	"""Enforce some standardization on metadata fields"""
# 	substitutions = dict(gender=dict(male=['m','man','male']),
# 	for k, v in user_info.items():
# 		this_value = v.lower().strip(' ').strip('"')
# 		# height
# 		if k=='height':
# 			if "'" in v:
# 				# height given in ft'in"

# 			if v > 90:

def write_user_info(fname, metadata):
	"""Write user info (metadata) to file"""
	with open(fname, mode='w') as fid:
		fid.write('key,value\n')
		for k, v in metadata.items():
			if v is None:
				vv = ''
			else:
				vv = v
			fid.write(f'{k},{vv}\n')

### --- GPS parsing --- ###
syntax = OrderedDict(**{
  1:  ['gps', 'lat', 'lon', 'alt'],	 # deg, deg, meters MSL WGS84
  3:  ['accel', 'x', 'y', 'z'],		 # m/s/s
  4:  ['gyro', 'x', 'y', 'z'],		  # rad/s
  5:  ['mag', 'x', 'y', 'z'],		   # microTesla
  6:  ['gpscart', 'x', 'y', 'z'],	   # (Cartesian XYZ) meters
  7:  ['gpsv', 'x', 'y', 'z'],		  # m/s
  8:  ['gpstime', ''],				  # ms
  81: ['orientation', 'x', 'y', 'z'],   # degrees
  82: ['lin_acc',	 'x', 'y', 'z'],
  83: ['gravity',	 'x', 'y', 'z'],   # m/s/s
  84: ['rotation',	'x', 'y', 'z'],   # radians
  85: ['pressure',	''],			  # ???
  86: ['battemp', ''],				  # centigrade
# Not exactly sensors, but still useful data channels:
 -10: ['systime', ''],
 -11: ['from', 'IP', 'port'],
})

index_to_column = OrderedDict(**{
  1:  [1, 2, 3],	 # deg, deg, meters MSL WGS84
  3:  [4, 5, 6],	 # m/s/s
  4:  [7, 8, 9],	 # rad/s
  5:  [10, 11, 12],  # microTesla
  6:  [13, 14, 15],  # (Cartesian XYZ) meters
  7:  [16, 17, 18],  # m/s
  8:  [19],		  # ms
  81: [20, 21, 22],  # degrees
  82: [23, 24, 25],
  83: [26, 27, 28],  # m/s/s
  84: [29, 30, 31],  # radians
  85: [32],		  # ???
  86: [33],		  # centigrade

# Not exactly sensors, but still useful data channels:
 -10: [34],
 -11: [35, 36],
})

column_keys = list(syntax.keys())
column_names = ['time']
for cname in syntax.values():
	if len(cname) == 2:
		cnames = [cname[0]]
	else:
		cnames = ['_'.join([cname[0], x]) for x in cname[1:]]
	column_names += cnames	

def _parse_sensorstream_line(line):
	"""Get contents of one line of gps.csv file"""
	out = np.full(len(column_names), np.nan)
	tmp = [x.strip() for x in line.split(',')]
	tmp = [np.float(x) if x is not '' else np.nan for x in tmp]
	out[0] = tmp.pop(0)
	while len(tmp) > 0:
		j = tmp.pop(0)
		if np.isnan(j):
			continue
		j = int(j)
		if not int(j) in column_keys:
			continue
		for jj in index_to_column[j]:
			value = tmp.pop(0)
			#print(jj, column_names[jj], value)
			out[jj] = value
	return out

def parse_sensorstream_gps(fname, sub_type):
	"""Parse gps file into 
	
	Parameters
	----------
	fname : string
		Description
	data_type : str
		string, optionally with ':' specifying a sub-component of gps data (e.g. latitude)
	
	Returns
	-------
	timestamps : arary
		array of timestamps for gps data. STILL NOT ON SAME TIME SCALE AS REST OF STREAMS.
		requires cross-correlation of inertial sensor data.
	gps_dict : OrderedDict
		dict of gps values. 

	Notes
	-----
	For each key, values will be removed for any values that are nans across
	all keys for a given time point. Time points will also be removed. Thus, 
	specifying different sub-components of 
	"""
	with open(fname) as fid:
		lines = fid.readlines()
	# Parse line by line
	values = np.vstack([_parse_sensorstream_line(line) for line in lines])
	tmp = OrderedDict((cn, val) for cn, val in zip(column_names, values.T))
	tt = tmp.pop('time')
	if sub_type is not None:
		# Parse component of gps
		key_list = [k for k in tmp.keys() if sub_type in k]
		out = OrderedDict((k, tmp[k]) for k in key_list)
	else:
		out = tmp
	# Parse for nans
	key_list = list(out.keys())
	value_present = np.vstack([~np.isnan(out[k]) for k in key_list]).T
	value_present = np.any(value_present, axis=1)
	tt = tt[value_present]
	for k in key_list:
		out[k] = out[k][value_present]

	return tt, out


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