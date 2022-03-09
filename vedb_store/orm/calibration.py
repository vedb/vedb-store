from .mappedclass import MappedClass
from .. import options, utils
import numpy as np
import copy
import os

BASE_PATH = options.config.get('paths', 'proc_directory')

class Calibration(MappedClass):
	def __init__(self, 
              	type='Calibration',
                calibration_class='vedb_gaze.calibration.Calibration',
                session=None,
				pupil_detection=None, 
				marker_detection=None, 
				eye=None,
				epoch=None,
				failed=None,
                params=None,
				data=None,
                tag=None,
				fname=None, 
				dbi=None,
				_id=None, 
				_rev=None):
		"""Computed calibration for from a session"""
		inpt = locals()
		self.type = 'Calibration'
		computed_defaults = ['data', 'fname', 'path']
		for k, v in inpt.items():
			if k in computed_defaults:
				setattr(self, '_' + k, v)
			elif not k in ['self', 'type',]:
				setattr(self, k, v)
		self._path = None
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = ['data']
		# Constructed on the fly and not saved to docdict
		self._temp_fields = ['path', 'calibration']
		# Fields that are other database objects
		self._db_fields = ['session', 'pupil_detection', 'marker_detection', 'params']

	def load(self,):
		"""Load computed calibration parameters

		Parameters
		----------
		type : str
			'arraydict', 'dictlist', or 'dataframe'

		"""
		# Fill fields that are database objects
		self.db_load()
		# Manage output type. Archival data is a dictionary of arrays; convert to desired output
		calibration_class = utils.get_function(self.calibration_class)
		self.calibration = calibration_class.load(self.fpath)
	
	def save(self, is_overwrite=False):
		if self.data is None:
			raise ValueError("Can't save without computed calibration data!")
		if self.dbi is None:
			raise ValueError("dbi (database interface) field must be specified to save object!")
		# Search for extant db object
		doc = self.db_fill(allow_multiple=False)
		if (doc._id is not None) and (doc._id in self.dbi.db) and (not is_overwrite):
			raise Exception(
				"Found extant doc in database, and is_overwrite is set to False!")
		# Assure path, _id
		if doc._id is None:
			doc._id = self.dbi.get_uuid()
		if not os.path.exists(self.path):
			gaze_path = os.path.join(BASE_PATH, 'gaze')
			if not os.path.exists(gaze_path):
				raise ValueError(f"Base path for processed gaze ({gaze_path}) not found!")
			else:
				print('Creating folder %s...'%self.path)
				os.makedirs(self.path)
		np.savez(doc.fpath, **self.data)
		# Save header info to database
		self.dbi.put_document(doc.docdict)
		return doc

	@property
	def data(self):
		if self._data is None:
			if os.path.exists(self.fpath):
				self._data=dict(np.load(self.fpath, allow_pickle=True))
			else:
				raise ValueError("No data loaded or found")
		return self._data

	@property
	def fname(self):
		if self._fname is None:
			self.db_load()
			if not np.any([x is None for x in [self._id, self.params, self.eye]]):
				self._fname = 'calibration-{}-{}-{}.npz'.format(
					self.params.fn.split('.')[-1], 
					self.params.tag, 
					self._id)
		return self._fname

	@property
	def path(self):
		if self._path is None:
			self.db_load()
			if self.session is not None:
				self._path = os.path.join(BASE_PATH, 'gaze', self.session.folder)
		return self._path
