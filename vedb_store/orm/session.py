from .mappedclass import MappedClass
from .subject import Subject
from .recording import Camera, GPS, RecordingSystem
from .. import options
import file_io
from collections import OrderedDict
import datetime
import warnings
import textwrap
import pathlib
import numpy as np
import yaml
import os

from ..utils import parse_sensorstream_gps, parse_vedb_metadata, SUBJECT_FIELDS, SESSION_FIELDS

BASE_PATH = options.config.get('paths', 'vedb_directory')


# Question: track data_available in database? 
# For feature extraction: there may be multiple versions and/or parameters that we would like to use to compute stuff. 
# e.g. for gaze data. How to track that?
# Separate database class for processed session, w/ param dicts and preprocessing sequences? 

# dbi field - need it, yes?

class Session(MappedClass):
	def __init__(self, 
			subject=None, 
			experimenter_id=None, 
			experiment=None,
			study_site=None, 
			date=None,
			instruction=None, 
			scene=None, 
			folder=None, 
			lighting=None, 
			weather=None, 
			data_available=None,
			recording_duration=None, 
			recording_system=None, 
			start_time=None, 
			type='Session', 
			dbi=None, 
			_id=None, 
			_rev=None):
		"""Class for a data collection session for vedb project
		start_time : float
			Start time is the common start time for all clocks. Necessary for syncronization of disparate 
			frame rates and start lags

		"""


		inpt = locals()
		self.type = 'Session'
		for k, v in inpt.items():
			if not k in ['self', 'type', 'start_time', 'recording_duration']:
				setattr(self, k, v)
		self._base_path = BASE_PATH
		self._paths = None
		self._features = None
		self._start_time = start_time
		self._recording_duration = recording_duration
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = ['paths', 'features', 'datetime']
		# Fields that are other database objects
		self._db_fields = ['recording_system', 'subject']

	def refresh(self):
		"""Update list of available data to load from path. WIP."""
		pass
	
	def load(self, data_type, time_idx=None, frame_idx=None, **kwargs):
		"""
		Parameters
		----------
		data_type : string
			one of: 'world_camera', 'tracking_camera', 'eye_camera', 
			some data types have multiple sub-components; these can be accessed
			with colon syntax, e.g. data_type='odometry:location'
		time_idx : tuple
			(start time, end time) in seconds since the start of the session.
			Note that some streams (e.g. world_camera) may not start right at
			0 seconds, because they take some time to get started
		frame_idx : tuple
			(start_frame, end_frame)
		kwargs : dict
			passed to relevant loading function. 
			Note that for movie data, you can reshape and change the color format
			by using e.g. `size=(90, 120)` or `color='gray'` (see file_io.load_mp4)
		"""
		# Input check
		if 'idx' in kwargs:
			# Backward compatibility
			warnings.warn('`idx` argument to load() has been deprecated - please use `time_idx` instead!')
			time_idx = kwargs.pop("idx")
		if (time_idx is not None) and (frame_idx is not None):
			raise ValueError("Please specify EITHER time_idx OR frame_idx, but not both!")
		if ':' in data_type:
			data_type, sub_type = data_type.split(':')
		else:
			sub_type = None
		if data_type == 'gps':
			tt, data = parse_sensorstream_gps(self.paths[data_type][1], sub_type)
			return tt, data
		else:
			tf, df = self.paths[data_type]
			tt = np.load(tf) - self.start_time
			if time_idx is not None:
				st, fin = time_idx
				ti = (tt > st) & (tt < fin)
				tt_clip = tt[ti]
				indices, = np.nonzero(ti)
				st_i, fin_i = indices[0], indices[-1]+1
			elif frame_idx is not None:
				st_i, fin_i = frame_idx
				tt_clip = tt[st_i:fin_i]
			else:
				raise ValueError('Please specify either `time_idx` or `frame_idx`')
			if 'odometry' in data_type:
				# Consider handling indices in load_msgpack; currently
				# an arg for idx is there, but not functional.
				dd = file_io.load_msgpack(df)
				dd = dd[st_i:fin_i]
				if sub_type is not None:
					dd = np.array([x[sub_type] for x in dd])
			else:
				dd = file_io.load_array(df, idx=(st_i, fin_i), **kwargs)
			return tt_clip, dd

	@property
	def start_time(self):
		if self._start_time is None:
			starts = []
			for k, v in self.paths.items():
				tfile, _ = v
				# look into memory mapping (e.g. kwarg `mmap_mode='r'`)
				# if this next line goes too slow
				try:
					# Kludge because first element of many timestamp 
					# arrays appears to be (spuriously) zero
					wait_frames = 1000
					tt = np.load(tfile)[:wait_frames]
					starts.append(np.min(tt[tt>0])) 
				except:
					print(f'Missing timestamps for {k}')
			self._start_time = np.min(starts)
		return self._start_time

	@property
	def recording_duration(self):
		"""Duration of recording in seconds"""
		if self._recording_duration is None:
			durations = []
			for k, v in self.paths.items():
				time_path, data_path = v
				if not os.path.exists(time_path):
					continue
				tt = np.load(time_path)
				stream_time = tt[-1] - self.start_time
				durations.append(stream_time)
			self._recording_duration = np.min(durations)
		return self._recording_duration
	@property
	def datetime(self):
		dt = datetime.datetime.strptime(self.date, '%Y_%m_%d_%H_%M_%S')
		return dt
	
	@property
	def paths(self):
		if self._paths is None:
			to_find = ['world.mp4', 'eye0.mp4', 'eye1.mp4', 't265.mp4', 'odometry.pldata', 'gps.csv'] # more?
			names = ['world_camera', 'eye_left', 'eye_right', 'tracking_camera', 'odometry', 'gps']
			_paths = {}
			for fnm, nm in zip(to_find, names):
				tt, ee = os.path.splitext(fnm)
				data_path = os.path.join(self._base_path, self.folder, fnm)
				data_path = self._resolve_sync_dir(data_path)
				timestamp_path = os.path.join(self._base_path, self.folder, tt + '_timestamps.npy')
				timestamp_path = self._resolve_sync_dir(timestamp_path)
				if os.path.exists(data_path):
					_paths[nm] = (timestamp_path, data_path)
			self._paths = _paths
		return self._paths
	
	@property
	def features(self):
		if self._features is None:
			# perhaps sort these by tag?
			self._features = self.dbi.query(type='Features', session=self._id)
		return self._features

	@classmethod
	def from_folder(cls, folder, dbinterface=None, raise_error=True, db_save=False, overwrite_yaml=False):
		"""Creates a new instance of this class from the given `docdict`.
		
		Parameters
		----------
		folder : string
			full path to folder name
		dbinterface : vedb-store.docdb.dbinterface
			db interface
		raise_error : bool
			Whether to raise an error if fields are missing. True simply raises an error, 
			False allows manual input of missing fields. 
		overwrite_yaml : bool
			Whether to create a new yaml file. Old yaml file will be saved as `config_orig.yaml`
			unless that file already exists (in which case no backup will be created - only
			original is backed up)


		"""
		ob = cls.__new__(cls)
		print('\n>>> Importing folder %s'%folder)
		# Crop '/' from end of folder if exists
		if folder[-1] == os.path.sep:
			folder = folder[:-1]
		base_dir, folder_toplevel = os.path.split(folder)
		# Catch relative path, make into absolute path
		if folder_toplevel == '':
			folder_toplevel = base_dir
			base_dir = ''
		if (len(base_dir) == 0) or (base_dir[0] != '/'):
			base_dir = os.path.abspath(os.path.join(os.path.curdir, base_dir))
		# Check for presence of folder in database if we are aiming to save session in database
		if db_save:
			check = dbinterface.query(type='Session', folder=folder_toplevel)
			if len(check) > 0:
				print('SESSION FOUND IN DATABASE.')
				return check[0]
			elif len(check) > 1:
				raise Exception('More than one database session found with this date!')				
		# Look for meta-data in folder
		yaml_file = os.path.join(folder, 'config.yaml')
		if not os.path.exists(yaml_file):
			raise ValueError('yaml file not found!')
		yaml_doc = yaml.load(open(yaml_file, mode='r'))
		# Get participant info if present
		participant_file = os.path.join(folder, 'user_info.csv')
		metadata = parse_vedb_metadata(yaml_file, participant_file, raise_error=raise_error, overwrite_yaml=overwrite_yaml)
		# Set date	(& be explicit about what constitutes date)	
		session_date = folder_toplevel
		try:
			# Assume year_month_day_hour_min_second for date specification in folder title
			dt = datetime.datetime.strptime(session_date, '%Y_%m_%d_%H_%M_%S')
		except:
			print('Date not parseable!')
		# Get subject, check for existence in database
		subject_params = dict((sf, metadata[sf]) for sf in SUBJECT_FIELDS if metadata[sf] is not None)
		if 'subject_id' not in subject_params:
			if 'experimenter_id' in metadata:
				e_id = 'experimenter_id=%s, '%metadata['experimenter_id']
			else:
				e_id = ''
			s_id = input("Enter subject_id (%s'abort' to quit): "%e_id)
			if s_id.lower() == 'abort':
				raise Exception('Manual quit')
			else:
				subject_params['subject_id'] = s_id
		subject = Subject(**subject_params, dbi=dbinterface)
		# Check for exisistence of this subject
		if dbinterface is not None:
			subject = subject.db_fill(allow_multiple=False)
		# If subject doesn't exist, save subject
		if subject._id is None:
			# Subject is not in database; display other subjects
			if db_save:
				print("Extant subjects w/ same subject_id:")
				other_subjects = dbinterface.query(type='Subject', subject_id=subject.subject_id)
				print(other_subjects)
				print("This subject:")
				print(subject.docdict)
				yn = input("Save subject? (y/n):")
				if yn.lower() in ['y', 't', '1']:
					subject = subject.save()
		else:
			print('Subject found in database!')

		def parse_resolution(res_string):
			out = res_string.strip('()').split(',')
			out = [int(x) for x in out]
			return out

		camera_types = ['world', 't265', 'eye0', 'eye1']
		camera_labels = ['world', 'tracking', 'eye_right', 'eye_left']
		cameras = {}
		for camera_type, camera_label in zip(camera_types, camera_labels):
			# Get camera parameters from yaml file
			cam_params = yaml_doc['streams']['video'][camera_type]
			recording_params = yaml_doc['commands']['record']['video'][camera_type]
			input_params = dict(
						manufacturer=cam_params['device_type'], 
						device_uid=str(cam_params['device_uid']),
						resolution=parse_resolution(cam_params['resolution']),
						fps=int(cam_params['fps']),
						codec=recording_params['codec'],
						crf=int(recording_params['encoder_kwargs']['crf']),
						preset=recording_params['encoder_kwargs']['preset'],
						color_format=cam_params['color_format'],
						)
			if camera_type == 'world':
				input_params['settings'] = cam_params['settings'] # includes exposure info
			# if 'eye' in camera_type:
			# 	input_params['controls'] = cam_params['controls'] # in theory, includes exposure info
			# Unclear what is best to do about above wrt profiles. 
			# Check for existence of this camera in database
			this_camera = Camera(dbi=dbinterface, **input_params)
			if dbinterface is not None:
				this_camera = this_camera.db_fill(allow_multiple=False)
			# If camera doesn't exist, save camera
			if this_camera._id is None:
				# Camera is not in database; offer to save if db_save is True
				if db_save:
					print("Extant cameras w/ same manufacturer:")
					other_cameras = dbinterface.query(type='Camera', manufacturer=this_camera.manufacturer)
					print(other_cameras)
					print("This camera:")
					print(this_camera.docdict)
					yn = input("Save camera? (y/n):")
					if yn.lower() in ['y', 't','1']:
						if camera_label == 'eye_left':
							default_tag = '{}_left_standard'.format(this_camera.manufacturer)
						elif camera_label == 'eye_right':
							default_tag = '{}_right_standard'.format(this_camera.manufacturer)
						else:
							default_tag = '{}_standard'.format(this_camera.manufacturer)
						tag = input("Please input tag for this camera [press enter for default: %s]:"%default_tag) or default_tag
						this_camera.tag = tag
						this_camera = this_camera.save()
			else:
				# Camera is in database
				print('%s camera found in database!'%camera_label)
			cameras[camera_label] = this_camera

		if os.path.exists(os.path.join(folder, 'gps.csv')):
			gps = 'phone'
		else:
			gps = None

		recording_system = RecordingSystem(world_camera=cameras['world'],
											eye_left=cameras['eye_left'],
											eye_right=cameras['eye_right'],
											tracking_camera=cameras['tracking'],
											tilt_angle=metadata['tilt_angle'],
											gps=gps, # UNDEFINED TO DO
											dbi=dbinterface,
											)
		# query for recording system in database
		if dbinterface is not None:
			recording_system = recording_system.db_fill(allow_multiple=False)
		if recording_system._id is None:
			# Recording system is not in database; give option to save it.
			if db_save:
				print("Extant recording systems:")
				other_systems = dbinterface.query(type='RecordingSystem')
				print(other_systems)
				print("This recording system:")
				print(recording_system.docdict)
				yn = input("Save recording_system? (y/n):")
				if yn.lower() in ['y', 't','1']:
					default_tag = 'vedb_standard'
					if recording_system.tilt_angle != 'unknown':
						default_tag = '_'.join([default_tag, '%d'%int(recording_system.tilt_angle)])
					tag = input("Please input tag for this recording_system [press enter for default: %s]:"%default_tag) or default_tag
					recording_system.tag = tag
					recording_system = recording_system.save()
		else:
			print('RecordingSystem found in database!')		
		
		# Define recording device, w/ tag
		params = dict((sf, metadata[sf]) for sf in SESSION_FIELDS)
		params['subject'] = subject
		params['folder'] = folder_toplevel
		params['date'] = session_date
		params['recording_system'] = recording_system
		print(params)

		ob.__init__(dbi=dbinterface, **params)
		# Temporarily set base directory to local base directory
		# This is a bit fraught.
		ob._base_path = base_dir
		return ob

	def __repr__(self):
		rstr = textwrap.dedent("""
			vedb_store.Session
			{d:>12s}: {date}
			{ss:>12s}: {study_site}/{experimenter_id}
			{r:>12s}: {recording_system}
			{i:>12s}: {instruction}
			{sc:>12s}: {scene}
			""")
		return rstr.format(
			d='date', 
			date=self.date,
			ss='collected by', 
			study_site=self.study_site, 
			experimenter_id=self.experimenter_id,
			r='system', 
			recording_system=self.recording_system.tag if isinstance(self.recording_system, MappedClass) else self.recording_system, 
			i='instruction',
			instruction=self.instruction,
			sc='scene',
			scene=self.scene,
			)

