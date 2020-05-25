from . import orm
from .orm.session import Session
from .orm.recording import RecordingSystem, RecordingDevice, Camera, Odometer, GPS
from .orm.segment import Segment

try:
    from .dbwrapper import docdb_lite as docdb
except ImportError:
    print('Failed to import docdb - no database functions available.')
