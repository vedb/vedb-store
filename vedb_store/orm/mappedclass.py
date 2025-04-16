# General database class for all vedb_store objects
import os
import six
import json
import warnings
import pathlib
import file_io as fio
from ..options import config


def _obj2id_strlist(value):
	if isinstance(value, MappedClass):
		# _id should always be top-level
		out = value._id
	elif isinstance(value, (list, tuple)):
		out = [_obj2id_strlist(v) for v in value]
	elif isinstance(value, dict):
		out = _obj2id_doc(value)
	else:
		out = value
	return out

def _obj2id_doc(doc):
	"""Map all database-mappable objects in a document (or dictionary) to database _ids

	searches through fields of a dict for strings, lists, or dictionaries in which 
	the values contain MappedClass objects
	"""
	out = {}
	for k, v in doc.items():
		if k[0] == '_' and not k in ('_id', '_rev'):
			# Avoid any keys in dict with "_" prefix
			continue
		else:
			if isinstance(v, (MappedClass, list, tuple)):
				# Separate function for lists / tuples
				out[k] = _obj2id_strlist(v)
			elif isinstance(v, dict):
				# Recursive call for dicts
				out[k] = _obj2id_doc(v)
			else:
				# For strings & anything but lists, tuples, and dicts, leave it alone
				out[k] = v 
	return out

def _id2obj_strlist(value, dbi):
	vb = dbi.is_verbose
	dbi.is_verbose = False
	v = value
	if isinstance(value, str):
		if value in ('None', 'multi_component'): 
			v = value
		else:
			v = dbi.query(1, _id=value, return_objects=True)
	elif isinstance(value, (list, tuple)):
		if len(value)>0:
			if isinstance(value[0], MappedClass): 
				# already loaded
				pass
			else:
				v = [dbi.query(1, _id=vv, return_objects=True) for vv in value]
	dbi.is_verbose = vb
	return v


class MappedClass(object):
	
	@property
	def docdict(self):
		return self._get_docdict()
	
	@property
	def datadict(self):
		return self._get_datadict()
	
	@property
	def fpath(self):
		if hasattr(self, 'path') and hasattr(self, 'fname'):
			if (self.path) is None or (self.fname is None):
				return None
			else:
				# # Deal with sync path
				path = self._resolve_sync_dir(self.path)
				if isinstance(self.fname, (list, tuple)):
					# Multi-file input. Get list of paths for all files.
					return [os.path.join(path, f) for f in self.fname]
				else:
					return os.path.join(path, self.fname)
		else:
			return None

	# @property
	# def _fname(self):
	# 	"""Default fname, overwrite in child classes if you want a real file name (e.g. <_id>.hdf)"""
	# 	return None

	def _resolve_sync_dir(self, path):
		if isinstance(path, pathlib.Path):
			path = str(path)
		# Deal with sync path
		sync_dir = config.get('paths', 'sync_directory')
		if sync_dir != '':
			# Sync dir is defined
			old, new = sync_dir.split(',')
			#print('Swapping out {} for {}'.format(old, new))
			path_out = path.replace(old, new)
			path_out = os.path.expanduser(path_out)
			# Make sure it's present
			if not os.path.exists(path_out):
				# This has proven really annoying, silencing. 
				# Best to do with some flag, but that's too complicated
				# for now.
				#warnings.warn('No sync dir found ({})'.format(path_out))
				path_out = os.path.expanduser(path)
		else:
			path_out = os.path.expanduser(path)
		return pathlib.Path(path_out)

	def _get_docdict(self, rm_fields=()):
		"""Get database header dictionary representation of this object

		Used to insert this object into a database or query a database for 
		the existence of this object. Maintains the option to remove some 
		fields. 
		"""
		# Remove fields that are never supposed to be saved in database
		_to_remove = ('docdict', 'datadict', 'fpath', 'dbi', 'data')
		attrs = [k for k in dir(self) if (not k[:2]=='__') and (not k in _to_remove)]
		attrs = [k for k in attrs if not callable(getattr(self, k))]
		# Do not save attributes with a value of None 
		no_value = [k for k in attrs if getattr(self, k) is None]
		# Exclusion criteria 
		to_remove = list(rm_fields) + no_value + self._data_fields + self._temp_fields
		# Get attribtues
		d = dict(((k, getattr(self, k)) for k in attrs if not k in to_remove))
		# Replace all classes in document with IDs
		d = _obj2id_doc(d)
		# Convert to json and back to avoid unpredictable behavior due to conversion, e.g. tuple!=list
		d = json.loads(json.dumps(d))
		return d

	def _get_stripped_docdict(self):
		dd = self._get_docdict()
		for key in ['fname', 'path', '_id', '_rev']:
			if key in dd:
				dd[key] = None
		return dd

	def _get_datadict(self, fields=None):
		if fields is None:
			fields = self._data_fields
		for f in fields:
			if hasattr(self, 'extras') and f in self.extras:
				raise Exception("There should be no data fields in 'extras' attribute!")
		dd = dict((k, self[k]) for k in fields if hasattr(self, k) and self[k] is not None)
		return dd

	def db_load(self, recursive=True):
		"""Load all attributes that are database-mapped objects objects from database.
		"""
		# (All of these fields better be populated by string database IDs)
		for dbf in self._db_fields: 
			v = getattr(self, dbf)
			if isinstance(v, (str, list, tuple)):
				v = _id2obj_strlist(v, self.dbi)
			elif v is None:
				pass
			elif isinstance(v, MappedClass):
				pass
			else:
				raise ValueError("You have a value in one of fields that is expected to be a db class that is NOT a dbclass or an ID or anything sensible. Noodle brain.")
			setattr(self, dbf, v)
		if recursive:
			for dbf in self._db_fields:
				if self[dbf] is None:
					# Allow missing database fields
					continue
				if isinstance(self[dbf], (list, tuple)):
					for item in self[dbf]:
						item.db_load(recursive=recursive)
				else:
					self[dbf].db_load(recursive=recursive)
		self._dbobjects_loaded = True

	def db_fill(self, skip_fields=('date_run', 'last_updated'), allow_multiple=False):
		"""Check database for identical instance.

		Parameters
		----------
		skip_fields : list or tuple
			fields to ignore in database check

		Returns
		-------
		docs : list of db dict(s)

		"""
		assert (not isinstance(self.dbi, dict)) and (not self.dbi is None)
		# Search for extant db object
		doc = self.docdict
		date_run = {}
		for sf in skip_fields:
			if sf in doc:
				if sf=='date_run':
					date_run[sf] = doc.pop(sf)
				else:
					_ = doc.pop(sf)
		chk = self.dbi.query_documents(**doc)
		if len(chk)==0:
			# Add date_run back in
			doc.update(date_run)
			return self.from_docdict(doc, self.dbi)
		elif len(chk)==1:
			# Fill doc with fields from chk
			doc.update(chk[0])
			# Add date_run back in, since we're now over-writing
			doc.update(date_run)
			return self.from_docdict(doc, self.dbi)
		else:
			# TODO: Add optional search narrowing here
			if allow_multiple:
				return [self.from_docdict(d, self.dbi) for d in doc]
			else:
				raise ValueError("Database object is not uniquely specified")
		
	def save(self, sdir=None, is_overwrite=False, data_transform=None, data_folder=None):
		"""Save the contents of this object to database
		
		Auto-generates a unique path in sdir (random uuid string)

		NOTE
		----
		We will need to deal with commented-out sections below when it comes time to save features
		"""
		# Initial checks
		assert self.dbi is not None, 'Must have database interface (`dbi`) property set in order to save!'
		# Search for extant db object
		doc = self.db_fill(allow_multiple=False)
		if (doc._id is not None) and (doc._id in self.dbi.db) and (not is_overwrite):
			raise Exception("Found extant doc in database, and is_overwrite is set to False!")
		# Assure path, _id
		if doc._id is None:
			doc._id = self.dbi.get_uuid()
		if len(self.datadict) > 0:
			fio.save_arrays(doc.fpath, meta=doc.docdict, **self.datadict)
		# Save header info to database
		self.dbi.put_document(doc.docdict)
		return doc

	def delete(self):
		assert (not isinstance(self.dbi, dict)) and (not self.dbi is None)
		if hasattr(self, 'path') and hasattr(self, 'fname'):
			if isinstance(self.fpath, (list, tuple)):
				# Leave listed files intact
				pass
			else:
				print('Deleting %s'%self.fpath)
				if not fio.fexists(self.fpath):
					raise Exception("Path to real file not found")
				fio.delete(self.fpath)
		else:
			raise Exception("Path to real file not found!")
		doc = self.dbi.db[self._id]
		self.dbi.db.delete(doc)
	
	@classmethod
	def from_docdict(cls, docdict, dbinterface):
		"""Creates a new instance of this class from the given `docdict`.
		"""
		ob = cls.__new__(cls)
		ob.__init__(dbi=dbinterface, **docdict)
		return ob

	### --- Housekeeping --- ###
	def __getitem__(self, x):
		return getattr(self, x)

	def __setitem__(self, x, y):
		return setattr(self, x, y)
	
