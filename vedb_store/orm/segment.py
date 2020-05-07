from .mappedclass import MappedClass
from .. import options
import file_io
import os

# Handling computed features:
# For session, automatically search for and store pointers to all feature spaces that have been computed for that session?
# Perhpas OK, with lazy loading of features_available...?

# Potential issue: there will be THOUSANDS, maybe tens or hundreds of thousands of these. Maybe don't store 
# every single (start, stop) in database? Maybe allow lists of indices? 
# Should indices be TIME values instead? Mappable in straightforward fashion to frames, depending on fps of desired data

# Label here may need to be broken into multiple kwargs. e.g. if we have a set list of human activities we want to label, 
# that could be one kw, and if we have eye events, that could be another...?
class Segment(MappedClass):
	def __init__(self, session=None, start_time=None, end_time=None, label=None, type='Segment', _id=None, _rev=None):
		"""Segment of data from a session"""
		inpt = locals()
		self.type = 'Segment'
		for k, v in inpt.items():
			if not k in ['self', 'type',]:
				setattr(self, k, v)
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = ['session']



	def load(self, data='world'):
		"""load something

		Parameters
		----------
		data : string or (class? class name?)
			if a string, loads raw data indicated by the string (e.g. 'world' 
			for world video, 'eye_left' for left eye video, etc). 
			could also be a class input
			class, loads 

		"""
		# Fill fields that are database objects
		self.db_load()
		data = file_io.load_array(self.session.paths[data_type], idx=(self.start_time, self.end_time))

	@classmethod
	def from_index(cls, index, session):
		# Generate object from 
		pass
