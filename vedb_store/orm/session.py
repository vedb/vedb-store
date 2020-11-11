from .mappedclass import MappedClass
from .subject import Subject
from .recording import Camera, GPS, Odometer, RecordingSystem
from .. import options
import file_io
import numpy as np
import yaml
import os

BASE_PATH = options.config.get('paths', 'vedb_directory')


# Question: track data_available in database? 
# For feature extraction: there may be multiple versions and/or parameters that we would like to use to compute stuff. 
# e.g. for gaze data. How to track that?
# Separate database class for processed session, w/ param dicts and preprocessing sequences? 

# dbi field - need it, yes?

class Session(MappedClass):
	def __init__(self, subject=None, experimenter=None, study_site=None, instruction=None, scene=None, folder=None, data_available=None,
		lighting=None, weather=None, 
		recording_duration=None, recording_system=None, start_time=None, type='Session', dbi=None, _id=None, _rev=None):
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
		self._paths = None
		self._features = None
		self._start_time = start_time
		self._recording_duration = recording_duration
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = ['paths', 'features']
		# Fields that are other database objects
		self._db_fields = ['recording_system', 'subject']

	def refresh(self):
		"""Update list of available data to load from path"""
		pass
	
	def load(self, data_type, idx=(0, 100), **kwargs):
		"""
		Parameters
		----------
		idx : tuple
			(start time, end time) in seconds (this is a TIME index for now!)
		"""
		if data_type in ('gps', 'odometry'):
			raise NotImplementedError("Note yet!") # Fix me!
		else:
			tf, df = self.paths[data_type]
			st, fin = idx
			tt = np.load(tf) - self.start_time
			ti = (tt > st) & (tt < fin)
			tt_clip = tt[ti]
			indices, = np.nonzero(ti)
			st_i, fin_i = indices[0], indices[-1]+1
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
	def paths(self):
		if self._paths is None:
			to_find = ['world.mp4', 'eye0.mp4', 'eye1.mp4', 't265.mp4', 'odometry.pldata', 'gps.csv'] # more?
			names = ['world_camera', 'eye_left', 'eye_right', 'tracking_camera', 'odometry', 'gps']
			_paths = {}
			for fnm, nm in zip(to_find, names):
				tt, ee = os.path.splitext(fnm)
				data_path = os.path.join(BASE_PATH, self.folder, fnm)
				data_path = self._resolve_sync_dir(data_path)
				timestamp_path = os.path.join(BASE_PATH, self.folder, tt + '_timestamps.npy')
				timestamp_path = self._resolve_sync_dir(timestamp_path)
				if os.path.exists(data_path):
					_paths[nm] = (timestamp_path, data_path)
			self._paths = _paths
		return self._paths
	
	@property
	def features(self):
		if self._features is None:
			# perhaps sort these by tag?
			self._features = self.dbi.query(type='FeatureSpace', session=self._id)
		return self._features
	@classmethod
	def from_folder(cls, folder, dbinterface):
		"""Creates a new instance of this class from the given `docdict`.
		"""
		ob = cls.__new__(cls)
		# Look for meta-data in folder
		yaml_file = os.path.join(folder, 'config.yaml')
		if not os.path.exists(yaml_file):
			raise ValueError('yaml file not found!')
		yaml_doc = yaml.load(yaml_file)
		# Check for fields in the yaml file
		session_fields = ['study_site',
							'experimenter_id',
							'lighting',
							'scene',
							'weather',
							'instruction',
							'tilt_angle',]
		subject_fields = ['subject_id',
							'age',
							'gender',
							'ethnicity',
							'IPD',
							'height',]
		required_fields = session_fields + subject_fields
		for field in required_fields:
			if not field in yaml_doc['metadata']:
				raise ValueError('Missing field %s'%field)
			# Get folder
			_, folder_toplevel = os.path.split(folder)
			# strftime call to parse folder name to date
			# ADD PARSING FUNCTION HERE
			session_date = folder_toplevel
		# Get subject, check for existence in database
		subject_params = dict((sf, yaml_doc[sf]) for sf in subject_fields)
		subject = Subject(**subject_params, dbi=dbi)
		# Check for exisistence of this subject!!
		# If subject doesn't exist, save subject
		def parse_resolution(res_string):
			out = res_string.strip('()').split(',')
			out = [int(x) for x in out]
			return out
		# Get world camera, check for existence in database
		wc_params = yaml_doc['streams']['video']['world']
		recording_params = yaml_doc['commands']['record']['video']['world']
		world_camera = Camera(
					manufacturer=wc_params['device_type'], 
					device_uid=wc_params['device_uid'],
					resolution=parse_resolution(wc_params['resolution']),
					fps=wc_params['fps'],
					codec=recording_params['codec'],
					crf=int(recording_params['encoder_kwargs']['crf']),
					preset=recording_params['encoder_kwargs']['preset'],
					color_format=wc_params['color_format'],
					settings=wc_params['settings'], # includes exposure info
					)
		# Check for existence of this camera in database
		# Ask for tag if not present
		# TO DO 

		# Get t265 camera, check for existence in database
		cam_params = yaml_doc['streams']['video']['t265']
		recording_params = yaml_doc['commands']['record']['video']['t265']
		tracking_camera = Camera(
					manufacturer=cam_params['device_type'], 
					device_uid=cam_params['device_uid'],
					resolution=parse_resolution(cam_params['resolution']),
					fps=cam_params['fps'],
					codec=recording_params['codec'],
					crf=int(recording_params['encoder_kwargs']['crf']),
					preset=recording_params['encoder_kwargs']['preset'],
					color_format=cam_params['color_format'],
					# May need to specify settings, may not
					#settings=cam_params['settings'], # includes exposure info
					)
		# Check for existence of this camera in database
		# Ask for tag if not present
		# TO DO 

		# Get t265 odometer, check for existence in database
		# TO DO 

		# Get eye cameras, check for existence in database
		cam_params = yaml_doc['streams']['video']['eye0']
		recording_params = yaml_doc['commands']['record']['video']['eye0']
		eye_right = Camera(
					manufacturer=cam_params['device_type'], 
					device_uid=cam_params['device_uid'],
					resolution=parse_resolution(cam_params['resolution']),
					fps=cam_params['fps'],
					codec=recording_params['codec'],
					crf=int(recording_params['encoder_kwargs']['crf']),
					preset=recording_params['encoder_kwargs']['preset'],
					color_format=cam_params['color_format'],
					# May need to specify settings, may not
					#settings=cam_params['settings'], # includes exposure info
					)
		# Check for existence of this camera in database
		# Ask for tag if not present
		# TO DO 		
		cam_params = yaml_doc['streams']['video']['eye1']
		recording_params = yaml_doc['commands']['record']['video']['eye1']
		eye_left = Camera(
					manufacturer=cam_params['device_type'], 
					device_uid=cam_params['device_uid'],
					resolution=parse_resolution(cam_params['resolution']),
					fps=cam_params['fps'],
					codec=recording_params['codec'],
					crf=int(recording_params['encoder_kwargs']['crf']),
					preset=recording_params['encoder_kwargs']['preset'],
					color_format=cam_params['color_format'],
					# May need to specify settings, may not
					#settings=cam_params['settings'], # includes exposure info
					)
		# Check for existence of this camera in database
		# Ask for tag if not present
		# TO DO

		recording_system = RecordingSystem(world_camera=world_camera,
											eye_left=eye_left,
											eye_right=eye_right,
											tracking_camera=tracking_camera,
											odometry=odometry, # UNDEFINED TO DO
											gps=gps, # UNDEFINED TO DO
											dbi=dbi,
											)
		# query for recording system in database
		# ask for tag if not present
		# TO DO 
		
		# Define recording device, w/ tag
		params = dict((sf, yaml_doc[sf]) for sf in session_fields)
		params['subject'] = subject
		params['dbi'] = dbi
		params['folder'] = folder_toplevel
		params['date'] = session_date
		recording_system = recording_system

		ob.__init__(dbi=dbinterface, **params)
		return ob



# e.g.:
# 		resolution: (1280, 1024) #(2048, 1536)
#       fps: 30
#       name: 'world'
#       codec: 'libx265'
#       device_type: flir
#       device_uid: FLIR_19238305
#       color_format: bgr24
