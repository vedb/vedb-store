# FeatureSpace class

import numpy as np
import json
import file_io as fio
from ..options import config
from .mappedclass import MappedClass
from ..utils import _is_numeric
import os


class ParamDictionary(MappedClass): 
    """Class representation of a feature space computed of a stimulus 
    """
    def __init__(self, type='ParamDictionary', fn=None, data_fields=None, tag='default_params', 
                 dbi=None, path=None, fname=None, _id=None, _rev=None, **params):
        """Class for parameter dictionary. Loads to/from database. 

        Parameters
        ----------
        path : string
            folder containing file
        fname : string
            file name (without absolute path)

        """
        # Internalize all inputs
        self.type = 'ParamDictionary' 
        # Flags
        self._dbobjects_loaded = False
        self._data_loaded = False
        # Introspection 
        if data_fields is None:
            self._data_fields = [k for k, v in params.items() if isinstance(v, np.ndarray)]
        else:
            self._data_fields = data_fields
        self._temp_fields = [] # ignored
        self._db_fields = [] # ignored... probably? 
        self.dbi = dbi
        self.tag = tag
        self.fn = fn
        self._fname = fname
        self._path = path
        self._id = _id
        self._rev = _rev
        self.params = params
        # Load data fields (automatic load could prove problematic)
        if self.fpath is not None:
            self.load()

    def _get_docdict(self, rm_fields=()):
        """Get docdb (database header) dictionary representation of this object

        NOTE: This class OVERWRITES the parent class function, since this is meant to only store 
        a parameter dictionary + a few other fields. 

        Used to insert this object into a docdb database or query a database for 
        the existence of this object.

        Maintain the option to remove some fields - this will be handy for partial copies
        of database objects 
        """
        # Cull data fields from param dict
        d = dict(_id=self._id,
                 _rev=self._rev,
                 type='ParamDictionary',
                 tag=self.tag,
                 fn=self.fn,
                 data_fields=self.data_fields,
                 fname=self.fname,
                 )
        
        no_value = [k for k in d.keys() if (getattr(self, k) is None) or (getattr(self, k)==[])]
        d = dict((k, v) for k, v in d.items() if not k in no_value)
        # Skip parameter values that are None (??)
        no_value_pp = [k for k in self.params.keys() if self.params[k] is None]
        skip_fields = self._data_fields + no_value_pp + list(rm_fields)
        pp = dict((k, v) for k, v in self.params.items() if not k in skip_fields)
        d.update(**pp) 
        d = json.loads(json.dumps(d))
        return d

    def _get_datadict(self):
        dd = dict((k, self.params[k]) for k in self._data_fields)
        return dd


    def load(self, cache_dir=None):
        """Load param dict into memory from docdb database
        docdict will have a param_dict element in it; 
        that will be updated with data fields
        """
        for k in self.data_fields:
            self.params[k] = fio.load_array(self.fpath, k, cache_dir=cache_dir)
        return self.params


    @property
    def path(self):
        if self._path is None:
            self._path = os.path.join(config.get('paths', 'vedb_directory'), 'processed', 'analysis_parameters')
        return self._path

    @property
    def fname(self):
        if self._fname is None:
            if not any([x is None for x in [self._id, self.fn, self.tag]]):
                _id = self._id
                param_tag = self.tag
                fn_name = self.fn.split('.')[-1]
                self._fname = f'{fn_name}-{param_tag}-{_id}.hdf'
        return self._fname

    def save(self, sdir=None, is_overwrite=False):
        """Save entry to database / save 
        """
        if sdir is not None:
            self._path = sdir
        if self._id is None:
            self._id = self.dbi.get_uuid()

        return super(ParamDictionary, self).save(sdir=sdir, is_overwrite=is_overwrite, data_folder='Parameters')


    @property
    def data_fields(self):
        return self._data_fields


    @classmethod
    def get_options(cls, fn, dbi):
        function_name = '.'.join([fn.__module__, fn.__name__])
        pds = dbi.query(type='ParamDictionary', fn=function_name)
        return sorted([p.tag for p in pds])


    ### ---- Housekeeping --- ###
    def __repr__(self):
        nm = '<vedb_store.ParamDictionary>\n'
        keys = ['fn', 'tag', 'params', 'path', 'fname', '_id', '_rev']
        args = ()
        for k in keys:
            if hasattr(self, k):
                to_add = repr(getattr(self, k))
                to_add = to_add.split('\n')[0]
                args+=(k, to_add)
        ss = '%10s : %s\n'*int(len(args)/2)
        ss = ss%args
        return nm+ss

