from .mappedclass import MappedClass
from .. import options, utils
import numpy as np
import os

BASE_PATH = options.config.get('paths', 'proc_directory')

class MarkerDetection(MappedClass):
	def __init__(self, 
              	type='MarkerDetection',
				data=None, 
				session=None, 
				marker_type=None,
				detection_params=None,
				epoch_params=None,
				epoch_bytype=None,
				failed=None,
				#epoch_overall=None, # TO DO.
				fname=None, 
				tag=None, 
				dbi=None,
				_id=None, 
				_rev=None):
		"""Detected markers for a session"""
		inpt = locals()
		self.type = 'MarkerDetection'
		computed_defaults = ['data', 'fname', ] # 'epoch_bytype'] # , 'epoch_overall']
		for k, v in inpt.items():
			if k in computed_defaults:
				setattr(self, '_' + k, v)
			elif not k in ['self', 'type',]:
				setattr(self, k, v)
		# Introspection
		# Will be written to self.fpath (if defined)
		self._data_fields = ['data']
		# Constructed on the fly and not saved to docdict
		self._temp_fields = ['timestamp', 'path']
		# Fields that are other database objects
		self._db_fields = ['session', 'detection_params', 'epoch_params']
		# Placeholder for computed timestamps
		self._path = None
		self._timestamp = None

	def load(self, type='arraydict'):
		"""Load labeled pupils

		Parameters
		----------
		type : str
			'arraydict', 'dictlist', or 'dataframe'

		"""
		# Fill fields that are database objects
		self.db_load()
		# Manage output type. Archival data is a dictionary of arrays; convert to desired output
		if type == 'arraydict':
			return self.data
		elif type=='dictlist':
			return utils.arraydict_to_dictlist(self.data)
		elif type=='dataframe':
			return utils.arraydict_to_dataframe(self.data, mapping=utils.mapping_marker_to_df)

	
	def save(self, is_overwrite=False):
		if self.data is None:
			raise ValueError("Can't save without data!")
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
		np.savez(doc.fpath, **self.data)
		# Save header info to database
		self.dbi.put_document(doc.docdict)
		return doc

	@property
	def timestamp(self):
		self.db_load()
		if self._timestamp is None:
			self._timestamp = self.data['timestamp'] - self.session.start_time
		return self._timestamp

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
		self.db_load()
		if self._fname is None:
			if not np.any([x is None for x in [self._id, self.detection_params, self.marker_type]]):
				epoch_str = 'all' if ((self.epoch_bytype is None) or (self.epoch_bytype == 'all')) else '%02d'%self.epoch_bytype
				self._fname = 'markers-{}-{}-{}-epoch{}-{}.npz'.format(
					self.marker_type,
					self.detection_params.fn.split('.')[-1], 
					self.detection_params.tag, 
					epoch_str,
					self._id)
		return self._fname

	@property
	def path(self):
		if self._path is None:
			self.db_load()
			if self.session is not None:
				self._path = os.path.join(BASE_PATH, 'gaze', self.session.folder)
		return self._path

	# Complication: We'd like to compute overall order of marker epochs, 
	# regardless of calibration / validation (the two can be used 
	# interchangeably). We'd like this to be a property of the object. 
	# However, calibration and validation won't have the same detection
	# parameters, so it will be tricky to find the right ones. Tabling
	# this for later. 

	# @property
	# def epoch_bytype(self):
	# 	self.db_load()
	# 	if self.dbi is None:
	# 		return None
	# 	epochs = self.dbi.query(type='MarkerDetection', 
	# 		session=self.session._id,
	# 		detection_params=self.detection_params._id,
	# 		epoch_params=self.epoch_params)