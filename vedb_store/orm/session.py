from .mappedclass import MappedClass
from .subject import Subject
from .recording import Camera, GPS, Odometer, RecordingSystem
from .. import options
import file_io
import datetime
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
	def __init__(self, 
                subject=None, 
                experimenter_id=None, 
                study_site=None, 
                date=None,
                instruction=None, 
                scene=None, 
                folder=None, 
		lighting=None, 
                weather=None, 
                tilt_angle=None,
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
			self._features = self.dbi.query(type='FeatureSpace', session=self._id)
		return self._features
	@classmethod
	def from_folder(cls, folder, dbinterface, raise_error=True):
		"""Creates a new instance of this class from the given `docdict`.
		"""
		ob = cls.__new__(cls)
		# Look for meta-data in folder
		yaml_file = os.path.join(folder, 'config.yaml')
		if not os.path.exists(yaml_file):
			raise ValueError('yaml file not found!')
		yaml_doc = yaml.load(open(yaml_file, mode='r'))
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
		if 'metadata' in yaml_doc:
			# Current version, good.
			metadata = yaml_doc['metadata']
		elif 'metadata' in yaml_doc['commands']['record']:
			# Legacy config compatibility
			metadata = yaml_doc['commands']['record']['metadata']
		else: 
			metadata = None
		if metadata is None:
			raise ValueError("Missing metadata in yaml file in folder.")
		missing_fields = list(set(required_fields) - set(metadata.keys()))
		if len(missing_fields) > 0:
			if raise_error: 
				raise ValueError('Missing fields: {}'.format(missing_fields))
			else:
				print('Missing fields: ', missing_fields)
		# Get folder
		base_dir, folder_toplevel = os.path.split(folder)
		# strftime call to parse folder name to date
		session_date = folder_toplevel # [:10] for date (YYYY_MM_DD) only; for now, keep time
		try:
			dt = datetime.datetime.strptime(session_date, '%Y_%m_%d_%H_%M_%S')
		except:
			print('Date not parseable!')
		# Get subject, check for existence in database
		subject_params = dict((sf, metadata[sf]) for sf in subject_fields)
		subject = Subject(**subject_params, dbi=dbinterface)
		# Check for exisistence of this subject
		subject = subject.db_fill(allow_multiple=False)
		# If subject doesn't exist, save subject
		if subject._id is None:
		    # Subject is not in database
		    # Display extant subjects with same subject_id:
		    print("Extant subjects w/ same subject_id:")
		    other_subjects = dbinterface.query(type='Subject', subject_id=subject.subject_id)
		    print(other_subjects)
		    print("This subject:")
		    print(subject.docdict)
		    yn = input("Save subject? (y/n):")
		    if yn.lower() in ['y', 't','1']:
		        subject.save()
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
			# Get world camera, check for existence in database
			cam_params = yaml_doc['streams']['video'][camera_type]
			recording_params = yaml_doc['commands']['record']['video'][camera_type]
			input_params = dict(
						manufacturer=cam_params['device_type'], 
						device_uid=cam_params['device_uid'],
						resolution=parse_resolution(cam_params['resolution']),
						fps=cam_params['fps'],
						codec=recording_params['codec'],
						crf=int(recording_params['encoder_kwargs']['crf']),
						preset=recording_params['encoder_kwargs']['preset'],
						color_format=cam_params['color_format'],
						)
			if camera_type == 'world':
				input_params['settings'] = cam_params['settings'] # includes exposure info
			this_camera = Camera(dbi=dbinterface, **input_params)
			# Check for existence of this camera in database
			this_camera = this_camera.db_fill(allow_multiple=False)
			# If camera doesn't exist, save camera
			if this_camera._id is None:
				# camera is not in database
				# Display extant world cameras with same manufacturer:
				print("Extant cameras w/ same manufacturer:")
				other_cameras = dbinterface.query(type='Camera', manufacturer=this_camera.manufacturer)
				print(other_cameras)
				print("This camera:")
				print(this_camera.docdict)
				yn = input("Save camera? (y/n):")
				if yn.lower() in ['y', 't','1']:
					if camera_type == 'eye_left':
						default_tag = '{}_left_standard'.format(this_camera.manufacturer)
					elif camera_type == 'eye_right':
						default_tag = '{}_right_standard'.format(this_camera.manufacturer)
					else:
						default_tag = '{}_standard'.format(this_camera.manufacturer)
					tag = input("Please input tag for this camera [press enter for default: %s]:"%default_tag) or default_tag
					this_camera.tag = tag
					this_camera.save()
			else:
				print('%s camera found in database!'%camera_type)
			cameras[camera_label] = this_camera

		# Get t265 odometer, check for existence in database
		odometry = None
		# GET GPS data if available
		gps = None
		# TO DO 

		recording_system = RecordingSystem(world_camera=cameras['world'],
											eye_left=cameras['eye_left'],
											eye_right=cameras['eye_right'],
											tracking_camera=cameras['tracking'],
											odometry=odometry, # UNDEFINED TO DO
											gps=gps, # UNDEFINED TO DO
											dbi=dbinterface,
											)
		# query for recording system in database
		recording_system = recording_system.db_fill(allow_multiple=False)
		if recording_system._id is None:
			# Recording system is not in database
			print("Extant recording systems:")
			other_systems = dbinterface.query(type='RecordingSystem')
			print(other_systems)
			print("This camera:")
			print(recording_system.docdict)
			yn = input("Save recording_system? (y/n):")
			if yn.lower() in ['y', 't','1']:
				default_tag = 'vedb_standard'
				tag = input("Please input tag for this recording_system [press enter for default: %s]:"%default_tag) or default_tag
				recording_system.tag = tag
				recording_system.save()
		else:
			print('RecordingSystem found in database!')		
		
		# Define recording device, w/ tag
		params = dict((sf, metadata[sf]) for sf in session_fields)
		params['subject'] = subject
		params['folder'] = folder_toplevel
		params['date'] = session_date
		recording_system = recording_system

		ob.__init__(dbi=dbinterface, **params)
		# Temporarily set base directory to local base directory
		# This is a bit fraught.
		ob._base_path = base_dir
		return ob



# e.g.:
# 		resolution: (1280, 1024) #(2048, 1536)
#       fps: 30
#       name: 'world'
#       codec: 'libx265'
#       device_type: flir
#       device_uid: FLIR_19238305
#       color_format: bgr24
