"""vedb_store utility functions"""
from . import options
from collections import OrderedDict
import pathlib
import numpy as np
import yaml


BASE_PATH = options.config.get('paths', 'vedb_directory')
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

