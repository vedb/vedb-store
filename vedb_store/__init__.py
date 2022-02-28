from . import orm, options
from .orm.session import Session
from .orm.recording import RecordingSystem, RecordingDevice, Camera, Odometer, GPS
from .orm.segment import Segment
from .orm.subject import Subject
from .orm.paramdictionary import ParamDictionary
from .orm.pupil_detection import PupilDetection
from .orm.marker_detection import MarkerDetection
from .orm.calibration import Calibration

from functools import partial

try:
	from .dbwrapper import docdb_lite as docdb
	dbhost = options.config.get('db', 'dbhost')
	dbname = options.config.get('db', 'dbname')
	if (dbname is not None) and (dbname.lower() not in ('none', '')):
		docdb.getclient = partial(docdb.getclient, dbhost=dbhost, dbname=dbname)
except ImportError:
	print('Failed to import docdb - no database functions available.')
