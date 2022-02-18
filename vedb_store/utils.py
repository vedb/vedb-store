"""vedb_store utility functions"""
from . import options
from collections import OrderedDict
import pathlib
import numpy as np
import yaml


SESSION_FIELDS = ['study_site',
					'experimenter_id',
					'lighting',
					'scene',
					'weather',
					'instruction',]
SUBJECT_FIELDS = ['subject_id',
					'age',
					'gender',
					'ethnicity',
					'IPD',
					'height',]
RECORDING_FIELDS = ['tilt_angle',]

def parse_vedb_metadata(yaml_file, participant_file, raise_error=True, overwrite_yaml=False):
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
	overwrite_yaml : bool
		Whether to create a new yaml file. Old yaml file will be saved as `config_orig.yaml`
		unless that file already exists (in which case no backup will be created - only
		original is backed up)

	"""
	# Assure yaml_file is a pathlib.Path
	yaml_file = pathlib.Path(yaml_file)
	with open(yaml_file, mode='r') as fid:
		yaml_doc = yaml.load(fid)
	required_fields = SESSION_FIELDS + SUBJECT_FIELDS + RECORDING_FIELDS
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
			raise ValueError("Missing metadata in yaml file in folder.")
		else:
			metadata = {}
	user_info = parse_user_info(participant_file)
	metadata.update(user_info)
	metadata_keys = list(metadata.keys())
	metadata_keys = [k for k in metadata_keys if (metadata[k] is not None) and metadata[k] != '']
	missing_fields = list(set(required_fields) - set(metadata_keys))
	if len(missing_fields) > 0:
		if raise_error: 
			raise ValueError('Missing fields: {}'.format(missing_fields))
		else:
			# Fill in missing fields
			print('Missing fields: ', missing_fields)
			for mf in missing_fields:
				if (mf in SUBJECT_FIELDS):
					default = 'None'
				elif mf == 'tilt_angle':
					default = '100'
				else:
					default = 'unknown'

				value = input("Enter value for %s [press enter for '%s', type 'abort' to quit]"%(mf, default)) or default
				if value.lower() in ('abort' "'abort'"):
					raise Exception('Manual quit.')
				elif value.lower() in ('none'):
					value = None
				if mf in ('tilt_angle',):
					value = int(value)
				elif mf in ('age', 'height'):
					if value is not None and value not in ('unknown', 'None'):
						value = int(value)
				metadata[mf] = value
		if overwrite_yaml:
			# Optionally replace yaml file with new one
			if metadata_location == 'base':
				yaml_doc['metadata'] = metadata
			elif metadata_location == 'commands/record':
				yaml_doc['commands']['record']['metadata'] = metadata			
			new_yaml_file = yaml_file.parent / 'config_orig.yaml'
			if new_yaml_file.exists():
				# Get rid of new one, original is already saved
				yaml_file.unlink()			
			else:
				# Create backup
				print('copying to %s'%new_yaml_file)
				yaml_file.rename(new_yaml_file)			
			with open(yaml_file, mode='w') as fid:
				yaml.dump(yaml_doc, fid)
	return metadata


def parse_user_info(fname):
	"""Parse user_info csv file for session"""
	if fname is None:
		# Is it a good idea to just return an empty dict? Should we throw an error?
		return {}
	with open(fname) as fid:
		lines = fid.readlines()
		out = dict(tuple([y.strip() for y in x.split(',')]) for x in lines)
		for k in out.keys():
			if k in ['IPD']:
				try:
					out[k] = float(out[k])
				except ValueError:
					out[k] = None
			elif k in ['height', 'age']:
				try:
					out[k] = int(out[k])
				except ValueError:
					out[k] = None
	_ = out.pop('key')
	return out

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
