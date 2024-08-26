"""Class for subject info"""

from .mappedclass import MappedClass
from .. import options
import file_io
import textwrap
import numpy as np
import warnings
import yaml
import os

BASE_PATH = options.config.get('paths', 'vedb_directory')


# Question: track data_available in database? 
# For feature extraction: there may be multiple versions and/or parameters that we would like to use to compute stuff. 
# e.g. for gaze data. How to track that?
# Separate database class for processed session, w/ param dicts and preprocessing sequences? 

# dbi field - need it, yes?

class Subject(MappedClass):
	def __init__(self, subject_id=None, birth_year=None, gender=None, ethnicity=None, IPD=None, height=None,
		type='Subject', dbi=None, age=None, _id=None, _rev=None):
		"""Class for a data collection session for vedb project
		start_time : float
			Start time is the common start time for all clocks. Necessary for syncronization of disparate 
			frame rates and start lags

		"""


		inpt = locals()
		self.type = 'Subject'
		for k, v in inpt.items():
			if k == 'age':
				warnings.warn("`age` field is deprecated; use `birth_year` instead!")
			if not k in ['self', 'type']:
				setattr(self, k, v)
			
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = []
		# Constructed on the fly and not saved to docdict
		self._temp_fields = []
		# Fields that are other database objects
		self._db_fields = []
	
	def __repr__(self):
		rstr = textwrap.dedent("""
			vedb_store.Subject
			{id:>12s}: {subject_id}
			{demo:>12s}: birth year={birth_year}, gender={gender}, height={height}
			""")
		return rstr.format(
			id='identifier', 
			subject_id=self.subject_id, 
			demo='demographics', 
			birth_year=self.birth_year,
			gender=self.gender,
			height=self.height,
			)