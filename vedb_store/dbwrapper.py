# Wrapper for docdb classes
from .orm.session import Session #, Labels? (subclass Stimulus to allow for non-image/auditory/whatever inputs?)
from .orm.recording import RecordingSystem, RecordingDevice, Camera, Odometer, GPS
from .orm.segment import Segment
from .orm.subject import Subject
from .orm.paramdictionary import ParamDictionary
from .orm.pupil_detection import PupilDetection
from .orm.marker_detection import MarkerDetection
from .orm.calibration import Calibration

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
		ParamDictionary=ParamDictionary,
		PupilDetection=PupilDetection,
		MarkerDetection=MarkerDetection,
		Calibration=Calibration,
		)
except ImportError:
	print("Could not initialize classes in docdb database")
