from .mappedclass import MappedClass
from .. import options
import file_io
import os

class RecordingSystem(MappedClass):
	def __init__(self, type='RecordingSystem', tag=None, world_camera=None, eye_left=None, eye_right=None, tracking_camera=None, 
		odometry=None, gps=None, dbi=None, _id=None, _rev=None):
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
		self.odometry = odometry
		self.gps = gps
		
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = ['world_camera', 'eye_left', 'eye_right', 'tracking_camera', 'odometry', 'gps']


class RecordingDevice(MappedClass):
	def __init__(self, type='RecordingDevice', manufacturer=None, tag=None, fps=None, dbi=None, _id=None, _rev=None): # manufacturer ? 
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
	def __init__(self, type='Camera', manufacturer=None, tag=None, resolution=None, fps=None, name=None, codec=None, device_type=None, 
			device_uid=None, exposure=None, crf=None, color_format=None, dbi=None, _id=None, _rev=None):
		"""
		Parameters
		----------
		resolution : list
			[horizontal_dim, vertical_dim]
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
