from .mappedclass import MappedClass
from .. import options, utils
import textwrap
import numpy as np
import copy
import os

BASE_PATH = options.config.get('paths', 'proc_directory')

# Handling computed features:
# For session, automatically search for and store pointers to all feature spaces that have been computed for that session?
# Perhpas OK, with lazy loading of features_available...?

# Potential issue: there will be THOUSANDS, maybe tens or hundreds of thousands of these. Maybe don't store 
# every single (start, stop) in database? Maybe allow lists of indices? 
# Should indices be TIME values instead? Mappable in straightforward fashion to frames, depending on fps of desired data

# Label here may need to be broken into multiple kwargs. e.g. if we have a set list of human activities we want to label, 
# that could be one kw, and if we have eye events, that could be another...?
class PupilDetection(MappedClass):
	def __init__(self, 
              	type='PupilDetection',
				data=None, 
				session=None, 
				eye=None,
				failed=None,
				fname=None, 
				params=None, 
				tag=None, 
				dbi=None,
				_id=None, 
				_rev=None):
		"""Segment of data from a session"""
		inpt = locals()
		self.type = 'PupilDetection'
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
		self._temp_fields = ['timestamp','path']
		# Fields that are other database objects
		self._db_fields = ['session', 'params']
		# Placeholder for computed timestamps
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
			return copy.deepcopy(self.data)
		elif type=='dictlist':
			return utils.arraydict_to_dictlist(self.data)
		elif type=='dataframe':
			return utils.arraydict_to_dataframe(self.data, mapping=utils.mapping_pupil_to_df)

	
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
	def timestamp(self):
		self.db_load()
		if self._timestamp is None:
			self._timestamp = self.data['timestamp'] - self.session.start_time
		return self._timestamp

	@property
	def data(self):
		if self._data is None:
			if os.path.exists(self.fpath):
				self._data=dict(np.load(self.fpath, allow_pickle=-True))
			else:
				raise ValueError("No data loaded or found")
		return self._data

	@property
	def fname(self):
		self.db_load()
		if self._fname is None:
			if not np.any([x is None for x in [self._id, self.params, self.eye]]):
				self._fname = 'pupil_detection-{}-{}-{}-{}.npz'.format(
					self.eye,
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

	def __repr__(self):
		self.db_load()
		rstr = textwrap.dedent("""
			vedb_store.PupilDetection
			{d:>12s}: {date}
			{t:>12s}: {tag}
			{e:>12s}: {eye}

			""")
		return rstr.format(
                    d='date',
                    date=self.session.folder,
                    t='tag',
                    tag=self.tag,
                    e='eye',
                    eye=self.eye,
                )
