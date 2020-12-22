from .mappedclass import MappedClass
from .. import options
import file_io
import os

class RecordingSystem(MappedClass):
	def __init__(self, type='RecordingSystem', 
		tag=None, 
		world_camera=None, 
		eye_left=None, 
		eye_right=None, 
		tracking_camera=None,
		tilt_angle=None,
		odometry=None, 
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
		odometry : RecordingDevice instance
			Odometry recording device & settings
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
		self.odometry = odometry
		self.gps = gps
		self._id = _id
		self._rev = _rev
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = ['world_camera', 'eye_left', 'eye_right', 'tracking_camera', 'odometry', 'gps']


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
