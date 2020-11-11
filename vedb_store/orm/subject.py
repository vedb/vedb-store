"""Class for subject info"""

from .mappedclass import MappedClass
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

class Subject(MappedClass):
	def __init__(self, subject_id=None, age=None, gender=None, ethnicity=None, ipd=None, height=None,
		type='Subject', dbi=None, _id=None, _rev=None):
		"""Class for a data collection session for vedb project
		start_time : float
			Start time is the common start time for all clocks. Necessary for syncronization of disparate 
			frame rates and start lags

		"""


		inpt = locals()
		self.type = 'Subject'
		for k, v in inpt.items():
			if not k in ['self', 'type']:
				setattr(self, k, v)
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []
	