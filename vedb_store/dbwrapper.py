# Wrapper for docdb classes
from .orm.mappedclass import MappedClass
from .orm.session import Session #, Labels? (subclass Stimulus to allow for non-image/auditory/whatever inputs?)
from .orm.recording import RecordingSystem, RecordingDevice, Camera, Odometer, GPS
from .orm.segment import Segment
from .orm.subject import Subject

try: 
    import docdb_lite
    docdb_lite.is_verbose = False
    docdb_lite.orm.class_type.update(
        Session=Session,
        Segment=Segment,
        RecordingSystem=RecordingSystem,
        RecordingDevice=RecordingDevice,
        Camera=Camera,
        GPS=GPS,
        Odometer=Odometer,
        Subject=Subject,
        )
except ImportError:
    print("Could not initialize classes in docdb database")
