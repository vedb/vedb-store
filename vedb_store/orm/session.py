from .mappedclass import MappedClass
from .. import options
import file_io
import numpy as np
import os

BASE_PATH = options.config.get('paths', 'vedb_directory')


# Question: track data_available in database? 
# For feature extraction: there may be multiple versions and/or parameters that we would like to use to compute stuff. 
# e.g. for gaze data. How to track that?
# Separate database class for processed session, w/ param dicts and preprocessing sequences? 

# dbi field - need it, yes?

class Session(MappedClass):
	def __init__(self, subject_id=None, subject_age=None, subject_gender=None, subject_ethnicity=None,
		experimenter=None, university=None, task=None, scene_category=None, folder=None, data_available=None,
		recording_duration=None, recording_system=None, start_time=None, type='Session', dbi=None, _id=None, _rev=None):
		"""Class for a data collection session for vedb project
		start_time : float
			Start time is the common start time for all clocks. Necessary for syncronization of disparate 
			frame rates and start lags

		"""


		inpt = locals()
		self.type = 'Session'
		for k, v in inpt.items():
			if not k in ['self', 'type', 'start_time']:
				setattr(self, k, v)
		self._paths = None
		self._features = None
		self._start_time = start_time
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = ['paths', 'features']
		# Fields that are other database objects
		self._db_fields = ['recording_system']

	def refresh(self):
		"""Update list of available data to load from path"""
		pass
	
	def load(self, data_type, idx=(0, 100)):
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
			dd = file_io.load_array(df, idx=(st_i, fin_i))
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
					tt = np.load(tfile)[:100]
					starts.append(np.min(tt[tt>0])) 
				except:
					print(f'Missing timestamps for {k}')
			self._start_time = np.min(starts)
		return self._start_time
	
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



# e.g.:
# 		resolution: (1280, 1024) #(2048, 1536)
#       fps: 30
#       name: 'world'
#       codec: 'libx265'
#       device_type: flir
#       device_uid: FLIR_19238305
#       color_format: bgr24