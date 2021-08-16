from .mappedclass import MappedClass
from .. import options
import file_io
import textwrap
import os

class RecordingSystem(MappedClass):
	def __init__(self, type='RecordingSystem', 
		tag=None, 
		world_camera=None, 
		eye_left=None, 
		eye_right=None, 
		tracking_camera=None,
		tilt_angle=None,
		rig_version=None,
		gps=None, 
		dbi=None, 
		_id=None, 
		_rev=None):
		"""Class to store the components and settings of all devices used to collect the data
		
		Parameters
		----------
		tag : str
			shorthand label for this RecordingSystem for easy retrieval
		world_camera : Camera instance
			world camera & settings
		eye_left : Camera instance
			eye camera & settings
		eye_right : Camera instance
			eye camera & settings
		tracking_camera : Camera instance
			tracking camera & settings
		tilt_angle : int
		        angle of mount between tracking camera and world camera
		gps : RecordingDevice instance
			GPS recording device & settings
		"""
		self.tag = tag
		self.dbi = dbi
		self.type = 'RecordingSystem'
		self.world_camera = world_camera
		self.eye_left = eye_left
		self.eye_right = eye_right
		self.tracking_camera = tracking_camera
		self.tilt_angle = tilt_angle
		self.rig_version = rig_version
		self.gps = gps
		self._dbobjects_loaded = all([isinstance(x, MappedClass) for x in [self.world_camera, self.eye_left, self.eye_right, self.tracking_camera]]) # later: all([isinstance(self.getattr(x), MappedClass) for x in self._db_fields])
		self._id = _id
		self._rev = _rev
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = ['world_camera', 'eye_left', 'eye_right', 'tracking_camera'] #, 'gps']
	def __repr__(self):
		if not self._dbobjects_loaded:
			try:
				self.db_load()
			except:
				pass
		try:
			rstr = textwrap.dedent("""
				vedb_store.RecordingSystem (tag: {tag})
				{w:>16s}: {world} - {w_uid}
				{t:>16s}: {tracking} - {t_uid}
				{e0:>16s}: {eye_right} - {e0_uid}
				{e1:>16s}: {eye_left} - {e1_uid}
				{a:>16s}: {tilt}
				{r:>16s}: {rig_version}
				""")[1:] # 1: to get rid of initial newline
			rstr = rstr.format(
				tag=self.tag,
				w='world_camera',
				world=self.world_camera.tag,
				w_uid=self.world_camera.device_uid,
				t='tracking_camera',
				tracking=self.tracking_camera.tag,
				t_uid=self.tracking_camera.device_uid,
				e0='eye_right',
				eye_right=self.eye_right.tag,
				e0_uid=self.eye_right.device_uid,
				e1='eye_left', 
				eye_left=self.eye_left.tag,
				e1_uid=self.eye_left.device_uid,
				a='tilt',
				tilt=self.tilt_angle,
				r='rig_version',
				rig_version=self.rig_version,
				)
			return rstr
		except:
			return 'vedb_store.RecordingSystem (no database fields available)'

class RecordingDevice(MappedClass):
	def __init__(self, type='RecordingDevice', 
		tag=None,
		manufacturer=None,
		name=None,
		device_uid=None,
		fps=None, 
		dbi=None, 
		_id=None, 
		_rev=None):
		"""Parent class for other recording devices. Inheritance is not used yet; perhaps delete"""
		inpt = locals()
		self.type = 'RecordingDevice'
		for k, v in inpt.items():
			if not k in ['self', 'type',]:
				setattr(self, k, v)

		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []


class Camera(RecordingDevice):
	def __init__(self, type='Camera', 
		tag=None, 
		manufacturer=None, 
		name=None,
		device_uid=None,
		resolution=None, 
		fps=None,
		codec=None, 
		crf=None,
		preset=None,
		settings=None,
		color_format=None,
		# More camera properties here
		dbi=None,
		_id=None,
		_rev=None):
		"""Camera recording device

		Parameters
		----------
		tag : str
			shorthand retrieval tag to specify this object. Intended to be more human-readable
			than the `_id` field, which absolutely must be unique; tag can potentially be
			the same for multiple objects (tho this is not advised)
		manufacturer : str
			Device manufacturer name (e.g. FLIR, Intel)
		name : str
			Device name (e.g. 'Chameleon', 'RealSense t265')
		device_uid : str
			Specifies serial number for a unique device (e.g. one particular FLIR world camera)
		resolution : list
			[horizontal_dim, vertical_dim] of video
		fps : int
			Frame rate for camera (desired - note this may be an approximate frame rate 
			depending on other camera settings)
		codec : str
			Encoding used to record video
		crf : str
			Compression factor if h264 encoding is used
		settings : dict
			dict specifying exposure settings and other parameters
		color_format : str
			Color format of video, e.g. 'RGB24', 'BGR24', etc
		dbi : str
			Database interface object, necessary for saving this object and for 
			querying database for other objects from this object
		_id : str
			Unique database identifier
		_rev : str
			Unique revision number for this object in the database			

		"""
		inpt = locals()
		self.type = 'Camera'
		for k, v in inpt.items():
			if not k in ['self', 'type',]:
				setattr(self, k, v)

		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []

	def __repr__(self):
		rstr = textwrap.dedent("""
			vedb_store.Camera (tag: {tag})
			{manufacturer} : {id} : [{x}, {y}] : {fps} fps
			""")[1:] # 1: to get rid of initial newline
		return rstr.format(
			tag=self.tag,
			manufacturer=self.manufacturer,
			id=self.device_uid,
			x=self.resolution[0],
			y=self.resolution[1],
			fps=self.fps)

class Odometer(RecordingDevice):
	def __init__(self, type='Odometer', 
		tag=None,
		manufacturer=None,
		name=None, 
		device_uid=None,
		dbi=None,
		_id=None,
		_rev=None,
		# More odometer properties here; SLAM version?
		):
		"""Class to save odometer properties
		Parameters
		----------
		tag : str
			shorthand retrieval tag to specify this object. Intended to be more human-readable
			than the `_id` field, which absolutely must be unique; tag can potentially be
			the same for multiple objects (tho this is not advised)
		manufacturer : str
			Device manufacturer name (e.g. FLIR, Intel)
		name : str
			Device name (e.g. 'Chameleon', 'RealSense t265')
		device_uid : str
			Specifies serial number for a unique device (e.g. one particular FLIR world camera)

		Notes
		-----
		fps does not seem to be saved / specifiable.
		May want to include version of SLAM algorithm that generated data?

		"""

		inpt = locals()
		self.type = 'Odometer'
		for k, v in inpt.items():
			if not k in ['self', 'type',]:
				setattr(self, k, v)

		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []
	

class GPS(RecordingDevice):
	def __init__(self, type='GPS', 
		tag=None,
		manufacturer=None,
		name=None, 
		device_uid=None,
		fps=None, 
		dbi=None,
		_id=None,
		_rev=None,
		# More GPS properties here
		):
		"""Class to save odometer properties
		Parameters
		----------
		tag : str
			shorthand retrieval tag to specify this object. Intended to be more human-readable
			than the `_id` field, which absolutely must be unique; tag can potentially be
			the same for multiple objects (tho this is not advised)
		manufacturer : str
			Device manufacturer name (e.g. FLIR, Intel)
		name : str
			Device name (e.g. 'Chameleon', 'RealSense t265')
		device_uid : str
			Specifies serial number for a unique device (e.g. one particular FLIR world camera)
		resolution : list
			[horizontal_dim, vertical_dim] of video
		fps : int
			Frame rate for GPS (may be approximate; unclear)
		"""
		inpt = locals()
		self.type = 'GPS'
		for k, v in inpt.items():
			if not k in ['self', 'type',]:
				setattr(self, k, v)

		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []
	
	def load(self, fpath, idx=(0, 100)):
		pass		
