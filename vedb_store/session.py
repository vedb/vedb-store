"""vedb_store.session
===================

Utilities for representing and working with VEDB recording sessions.

This module provides classes and helpers to manage session data and
temporal clips within sessions. Main classes and functions include

- Session
    Representation of a VEDB recording session. Methods load data
    streams, manage file paths, and interface with gaze pipelines.
- SessionClip
    Represents a labeled time interval within a session and provides
    methods to slice, sample, and visualize data for that interval.
- SessionClipList
    Container for multiple SessionClip instances WITHIN A SINGLE
    SESSION with logical operations and conversion utilities.
- ClipList
    Container for multiple SessionClip instances ACROSS sessions.
    These are each meant to constitute a sample of VEDB to be 
    analyzed. 

Notes
-----
Timestamps used throughout are typically zero-referenced to the
first frame of the world video (world_timestamps_0start.npy). Many
functions expect numpy arrays of timestamps and return time-indexed
slices of data.

Examples
--------
>>> from vedb_store import Session
>>> s = Session.from_folder('/path/to/session')
>>> tt, frames = s.load('world_camera', time_idx=(0, 5))
"""

from . import utils
from . import options
import file_io
import datetime
import warnings
import pathlib
import numpy as np
import copy
import os
import functools

from .utils import get_frame_indices, get_time_split, onoff_from_binary
# Gaze-pipeline file names and default processing tags come from vedb_gaze,
# so loading here always matches what vedb_gaze.pipelines.pipeline_vedb writes
from vedb_gaze.utils import make_file_strings, defaults as gaze_defaults
from vedb_gaze.options import config as gaze_config
from vedb_gaze.odometry import load_odometry


def _config_path(key, gaze_key=None):
    """Path from the vedb_store config `[paths]` section

    If the vedb_store value is empty (the default), fall back to `gaze_key`
    in the vedb_gaze config, so both packages use the same data folders.
    """
    value = options.config.get('paths', key, fallback='').strip()
    if (value == '') and (gaze_key is not None):
        value = gaze_config.get('paths', gaze_key, fallback='').strip()
    return pathlib.Path(value).expanduser()


BASE_PATH = _config_path('vedb_directory', 'base_dir')
PROC_PATH = _config_path('proc_directory', 'proc_dir')
SAMPLE_PATH = _config_path('sample_directory')
EYE_PRIVACY_SETTING = '' # '_blur'

# Tables of per-session metadata (in BASE_PATH), loaded on first use so that
# importing vedb_store does not require access to the data
_session_tables = {}


def _load_session_table(name):
    """Load (and cache) `BASE_PATH / <name>.npz` as a dict of arrays"""
    if name not in _session_tables:
        fpath = BASE_PATH / f'{name}.npz'
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} not found. Set `vedb_directory` in {options.usercfg} "
                                    "(or `base_dir` in the vedb_gaze config) to the folder with VEDB sessions.")
        _session_tables[name] = dict(np.load(fpath))
    return _session_tables[name]


def __getattr__(name):
    # Backward compatibility: SESSION_INFO / SESSION_ERROR* used to be loaded
    # at import; now they are loaded when first accessed
    if name == 'SESSION_INFO':
        return _load_session_table('session_info')
    if name == 'SESSION_ERROR':
        return _load_session_table('session_error')
    if name == 'SESSION_ERROR_LIST':
        return utils.arraydict_to_dictlist(_load_session_table('session_error'))
    if name == 'SESSION_ERROR_DICT':
        return dict((x['folder'], x) for x in utils.arraydict_to_dictlist(_load_session_table('session_error')))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation, patches


def _clip_arraydict(arraydict, index):
    """Apply `index` (slice or boolean array) along time to every field of
    `arraydict` that has one value per timestamp; other fields are dropped"""
    n = len(arraydict['timestamp'])
    return dict((k, v[index]) for k, v in arraydict.items()
                if isinstance(v, np.ndarray) and (np.ndim(v) > 0) and (len(v) == n))


def get_session_info(session, session_info=None):
    """Metadata (e.g. 'fov', 'val_4x7', 'val_7x9') for one session from session_info.npz"""
    if session_info is None:
        session_info = _load_session_table('session_info')
    tmp = utils.arraydict_to_dictlist(session_info)
    si_dict = dict((x['folder'], x) for x in tmp)
    si = si_dict[session]
    return si


def _session_file_kwargs(session):
    """Session-specific kwargs for `make_file_strings`, from session_info.npz

    Returns `fov_str` (e.g. 'fov101'), which replaces the fov token in the
    error tag, and `validation_checkerboard_size` ('4x7' or '7x9').
    """
    si = get_session_info(session)
    if si['val_4x7']:
        validation_checkerboard_size = '4x7'
    elif si['val_7x9']:
        validation_checkerboard_size = '7x9'
    else:
        raise ValueError('Could not figure out what size validation checkerboard was for this session!')
    return dict(fov_str=f'fov{si["fov"]:.0f}',
                validation_checkerboard_size=validation_checkerboard_size)

def show_ellipse(ellipse, img=None, ax=None, center_color='r', **kwargs):
    """Show an OpenCV-style ellipse on a Matplotlib axis, optionally with an image underlay.

    Parameters
    ----------
    ellipse : dict
        Ellipse parameters in OpenCV format with keys:
        - 'center' : tuple of float (x, y)
        - 'axes' : tuple of float (width, height) lengths of the ellipse axes
        - 'angle' : float, rotation angle in degrees
    img : array_like, optional
        Image to display as a background under the ellipse. If ``None``, no
        image is shown. Default is ``None``.
    ax : matplotlib.axes.Axes, optional
        Axes in which to plot. If ``None``, a new figure and axes are created.
    center_color : str or tuple, optional
        Color for the ellipse center marker. Default is ``'r'``.
    **kwargs : dict, optional
        Additional keyword arguments passed to ``matplotlib.patches.Ellipse``.

    Returns
    -------
    tuple
        ``(patch_h, pt_h)`` where ``patch_h`` is the :class:`matplotlib.patches.Ellipse`
        instance added to the axes and ``pt_h`` is the scatter handle for the
        ellipse center.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ell = patches.Ellipse(
        ellipse["center"], *ellipse["axes"], angle=ellipse["angle"], **kwargs
    )
    if img is not None:
        ax.imshow(img, cmap="gray")
    patch_h = ax.add_patch(ell)
    pt_h = ax.scatter(ellipse["center"][0], ellipse["center"][1], color=center_color)
    return patch_h, pt_h

def unique(seq, idfun=None): 
    """Returns only unique values in a list (with order preserved).
    (idfun can be defined to select particular values??)
    
    Stolen from the internets 11.29.11
    """
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: 
            seen[marker]+=1
            continue
        else:
            seen[marker] = 1
            result.append(item)         
    return result, seen

REQUIRED_FILES = [
    'accel.pldata',
    'accel_timestamps.npy',
    ('eye0.mp4','eye0_blur.mp4'),
    'eye0.pldata',
    'eye0_timestamps.npy',
    ('eye1.mp4', 'eye1_blur.mp4'),
    'eye1.pldata',
    'eye1_timestamps.npy',
    'gyro.pldata',
    'gyro_timestamps.npy',
    'odometry.pldata',
    'odometry_timestamps.npy',
    ('world.mp4', 'worldPrivate.mp4'),
    #'world.pldata',
    'world_timestamps.npy',
    ]


OPTIONAL_FILES = [
    'world.extrinsics',
    'world.intrinsics',
    'lens.json'
    ]

# Default processing tags for the gaze pipeline: the [defaults] section of
# the vedb_gaze config (keys 'pupil', 'eyelid', ..., 'calibration_epoch')
pipeline_default = gaze_defaults


def _get_pipeline(pipeline=None):
    """Full set of `make_file_strings` processing tags for `pipeline`

    Parameters
    ----------
    pipeline : dict or None
        processing tags, keyed as in `vedb_gaze.utils.make_file_strings`
        (e.g. 'pupil', 'calibration', 'error'). Tags not given are taken
        from `pipeline_default`. Old-style keys with a '_tag' suffix (e.g.
        'pupil_tag') are accepted, with a DeprecationWarning.
    """
    out = dict(pipeline_default)
    if pipeline is None:
        return out
    old_keys = [k for k in pipeline if k.endswith('_tag')]
    if len(old_keys) > 0:
        warnings.warn(f"Pipeline keys ending in '_tag' are deprecated ({old_keys}); "
                      "use vedb_gaze names, e.g. 'pupil' instead of 'pupil_tag'",
                      DeprecationWarning, stacklevel=3)
    for k, v in pipeline.items():
        k = k[:-len('_tag')] if k.endswith('_tag') else k
        if k not in out:
            raise ValueError(f"Unknown pipeline key '{k}'; valid keys: {list(out.keys())}")
        out[k] = v
    return out


class Session(object):
    """Representation of a VEDB recording session.
    
    Contains paths to all relevant files (world video, eye videos, etc.)
    and means to load them, as well as meta-data about the session.
    """
    def __init__(self, folder, clip_labels='native', clock='native', raise_error=True):
        """Class for a data collection session for vedb project

        """
        # Check for existence of folder
        if not isinstance(folder, pathlib.Path):
            folder = pathlib.Path(folder)
        # Check if folder is locally defined
        if folder.parent == pathlib.Path('.'):
            if folder.exists():
                folder = folder.absolute()
            else:
                folder = pathlib.Path(BASE_PATH) / folder.name
        if not folder.exists():
            raise ValueError(f"Folder {folder.name} not found!")
        # Check on files available in folder
        missing_files = _check_paths(folder)
        if (len(missing_files) > 0) & raise_error:
            raise ValueError(f'Missing files: {missing_files}\n')
        
        self.folder = folder.name
        self.clock = clock
        self.gaze_paths = None
        self._base_path = folder.parent
        self._path = None
        self._paths = None
        self._gaze = None
        self._pupil = None
        self._eyelids = None
        self._error = None
        self._features = None
        self._world_time = None
        self._recording_duration = self.world_time[-1] - self.world_time[0]
        if clip_labels is not None:
            self.load_clips(clip_labels)

    def load_clips(self, clip_labels='native'):
        clip_file = pathlib.Path(self.path) / f'{self.folder}.csv'
        self.clips = []
        if clip_file.exists():
            try:
                self.clips = parse_csv(clip_file)
            except Exception as e:
                warnings.warn(f"Can't parse clips in {clip_file}: {e}")
        else:
            warnings.warn(f"No clips defined for {self.folder} (no file {clip_file.name})")
        if clip_labels == 'native':
            clip_labels = ('native', 'native')
        loc_mapping, task_mapping = clip_labels
        # Location mapping
        if loc_mapping == 'native':
            pass
        else:
            loc_mapping = utils.read_yaml(BASE_PATH / f'{loc_mapping}.yaml')
        # Task mapping
        if task_mapping == 'native':
            pass
        else:
            task_mapping = utils.read_yaml(BASE_PATH / f'{task_mapping}.yaml')
        for j in range(len(self.clips)):
            if loc_mapping == 'native':
                location = self.clips[j].location
            else:
                location = loc_mapping[self.clips[j].location]
            if task_mapping == 'native':
                task = self.clips[j].task
            else:
                task = task_mapping[self.clips[j].task]
            self.clips[j].tag = f'{location}:{task}'
        

    def load_gaze_pipeline(self, pipeline=None, clock='native', validation_epoch=0, eye=None):
        """Set file names (`self.gaze_paths`) for a gaze pipeline

        File names are built by `vedb_gaze.utils.make_file_strings`, so they
        match the outputs of `vedb_gaze.pipelines.pipeline_vedb`.

        Parameters
        ----------
        pipeline : dict, optional
            dict of preprocessing step tags (strings), each of which identifies
            how a step of preprocessing was handled, keyed as in
            `vedb_gaze.utils.make_file_strings` (e.g. 'pupil', 'calibration',
            'error'). Tags not given are taken from `pipeline_default` (the
            vedb_gaze config defaults).
        clock : str, optional
            clock for loaded data; 'native' by default
        validation_epoch : int, optional
            validation epoch for the error file, by default 0
        eye : str or None, optional
            'left' or 'right' to fill in file names; None (default) leaves a
            '%s' placeholder for the eye
        """
        self.clock = clock
        self.gaze_paths = make_file_strings(**_get_pipeline(pipeline), **_session_file_kwargs(self.folder),
                                            eye=eye, validation_epoch=validation_epoch)


    def load_gaze(self, pipeline=None, eye='both', clock='native'):
        """Load estimate of gaze based on particular pipeline tag

        See also Session.gaze and Session.load_gaze_pipeline()

        Parameters
        ----------
        pipeline : dict, optional
            dict of preprocessing step tags (strings), keyed as in
            `vedb_gaze.utils.make_file_strings`; tags not given are taken
            from `pipeline_default` (see `Session.load_gaze_pipeline`)
        clock : str, optional
            which timestamps gaze should have. Default is 'native', which means at ~120 hz
            (native eye camera temporal resolution). 'world_time' specifies that gaze should be
            matched to nearest world timestamp (or averaged over a window according to
            `kwargs` that are passed on to `match_timepoints()`)
        """
        gaze_paths = make_file_strings(**_get_pipeline(pipeline), **_session_file_kwargs(self.folder), eye=None)

        gaze = dict(left=dict(np.load(PROC_PATH / self.folder / (gaze_paths['gaze']%'left'))),
                    right=dict(np.load(PROC_PATH / self.folder / (gaze_paths['gaze']%'right'))))
        # resample to `clock`
        if clock != 'native':
            if clock == 'world_time':
                this_time = self.world_time
            else:
                raise NotImplementedError("can only handle 'native' and 'world_time' for clocks so far")
            for e in gaze.keys():
                gaze[e] = utils.match_time_points(dict(timestamp=this_time), gaze[e])

        if eye == 'left':
            return gaze['left']
        elif eye == 'right':
            return gaze['right']
        elif eye == 'both':
            return gaze
        elif eye == 'best':
            if self.error['left']['gaze_err_weighted'] < self.error['right']['gaze_err_weighted']:
                return gaze['left']
            else:
                return gaze['right']
        return
    
    @property
    def gaze(self):
        """Always native eye video time for both eyes, if you want other clocks use Session.load_gaze()"""
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._gaze is None:
            self._gaze = dict((lr, dict(np.load(PROC_PATH / self.folder / (self.gaze_paths['gaze']%lr)))) \
                               for lr in ['left','right'])
        return self._gaze

    @property
    def error(self):
        """Gaze error for both eyes, for the validation epoch set in
        `load_gaze_pipeline()` (epoch 0 by default)"""
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._error is None:
            # make_file_strings returns a list of error files, one per
            # validation epoch; load_gaze_pipeline asks for a single epoch
            self._error = dict((lr, dict(np.load(PROC_PATH / self.folder / (self.gaze_paths['error'][0]%lr)))) \
                               for lr in ['left','right'])
        return self._error
    
    def load(self, data_type, time_idx=None, frame_idx=None, processed=True, **kwargs):
        """
        Parameters
        ----------
        data_type : string
            one of: 'world_camera', 'eye_left', 'eye_right', 'odometry'.
            some data types have multiple sub-components; these can be accessed
            with colon syntax, e.g. data_type='odometry:pitch'
        time_idx : tuple
            (start time, end time) in seconds since the start of the session.
            Note that some streams (e.g. world_camera) may not start right at
            0 seconds, because they take some time to get started
        frame_idx : tuple
            (start_frame, end_frame). For processed odometry, these index
            the resampled samples.
        processed : bool, optional
            odometry only. If True (default), load with
            `vedb_gaze.odometry.load_odometry` (resampled, smoothed, with
            'pitch', 'roll', 'yaw' in degrees); the data are a dict of
            arrays (or one array, for 'odometry:<field>'). If False, load
            the raw records from odometry.pldata (a list of dicts, or an
            array for 'odometry:<field>').
        kwargs : dict
            passed to relevant loading function (for processed odometry,
            to `load_odometry`, e.g. `resample`, `fps`, `smooth_kwargs`).
            Note that for movie data, you can reshape and change the color format
            by using e.g. `size=(90, 120)` or `color='gray'` (see file_io.load_mp4)

        Returns
        -------
        timestamps, data
        """
        # Input check
        if 'idx' in kwargs:
            # Backward compatibility
            warnings.warn('`idx` argument to load() has been deprecated - please use `time_idx` instead!')
            time_idx = kwargs.pop("idx")
        if (time_idx is not None) and (frame_idx is not None):
            raise ValueError("Please specify EITHER time_idx OR frame_idx, but not both!")
        if ':' in data_type:
            data_type, sub_type = data_type.split(':')
        else:
            sub_type = None
        if (data_type == 'odometry') and processed:
            odo = load_odometry(self.path, **kwargs)
            if time_idx is not None:
                st_i, fin_i = get_frame_indices(*time_idx, odo['timestamp'])
            elif frame_idx is not None:
                st_i, fin_i = frame_idx
            else:
                raise ValueError('Please specify either `time_idx` or `frame_idx`')
            odo = _clip_arraydict(odo, slice(st_i, fin_i))
            if sub_type is not None:
                return odo['timestamp'], odo[sub_type]
            return odo['timestamp'], odo
        tf, df = self.paths[data_type]
        tt = np.load(tf)
        if time_idx is not None:
            st, fin = time_idx
            st_i, fin_i = get_frame_indices(st, fin, tt)
            tt_clip = tt[st_i:fin_i]
        elif frame_idx is not None:
            st_i, fin_i = frame_idx
            tt_clip = tt[st_i:fin_i]
        else:
            raise ValueError('Please specify either `time_idx` or `frame_idx`')
        if 'odometry' in data_type:
            # Consider handling indices in load_msgpack; currently
            # an arg for idx is there, but not functional.
            dd = file_io.load_msgpack(df)
            dd = dd[st_i:fin_i]
            if sub_type is not None:
                dd = np.array([x[sub_type] for x in dd])
        else:
            dd = file_io.load_array(df, idx=(st_i, fin_i), **kwargs)
        return tt_clip, dd

    def get_video_handle(self, stream):
        """Return an opencv """
        return file_io.VideoCapture(str(self.paths[stream][1]))
    
    def get_video_time(self, stream):
        return np.load(self.paths[stream][0])

    def get_frame_indices(self, start_time, end_time, stream='world_camera'):
        all_time = self.get_video_time(stream) 
        return get_frame_indices(start_time, end_time, all_time)

    def check_paths(self, check_type='comprehensive'):
        """Return a list of missing files from the directory of this session"""
        return _check_paths(self.path)
            
    @property
    def world_time(self):
        if self._world_time is None:
            self._world_time = np.load(self.paths['world_camera'][0])
        return self._world_time
        
    @property
    def recording_duration(self):
        """Duration of recording in seconds"""
        if self._recording_duration is None:
            durations = []
            for k, v in self.paths.items():
                time_path, data_path = v
                if not os.path.exists(time_path):
                    continue
                tt = np.load(time_path)
                stream_time = tt[-1]
                durations.append(stream_time)
            self._recording_duration = np.min(durations)
        return self._recording_duration
        
    @property
    def datetime(self):
        if self.date is None:
            return None
        else:
            dt = datetime.datetime.strptime(self.date, '%Y_%m_%d_%H_%M_%S')
            return dt
    
    @property
    def path(self):
        if self._path is None:
            self._path = self._base_path / self.folder
        return self._path
        
    @property
    def paths(self):
        if self._paths is None:
            to_find = [('world.mp4','worldPrivate.mp4'),  ('eye1.mp4', 'eye1_blur.mp4'), ('eye0.mp4','eye0_blur.mp4'), 'odometry.pldata']
            names = ['world_camera', 'eye_left', 'eye_right', 'odometry']
            _paths = {}
            base_path = self.path # TODO: re-implement? self._resolve_sync_dir(self.path)
            for fnm, nm in zip(to_find, names):
                if isinstance(fnm, tuple):
                    ff = 'no_file.nope'
                    for f in fnm:
                        if (base_path / f).exists():
                            ff = copy.copy(f)
                            break
                else:
                    ff = copy.copy(fnm)
                fstem, _ = os.path.splitext(ff)
                if fstem == 'worldPrivate':
                    fstem = 'world'
                if 'blur' in fstem:
                    fstem = fstem[:4]
                data_path = base_path / ff
                timestamp_path = base_path / (fstem + '_timestamps_0start.npy')
                if data_path.exists() & timestamp_path.exists():
                    _paths[nm] = (timestamp_path, data_path)
                else:
                    print(f'failed for {nm}:', timestamp_path, data_path)
            self._paths = _paths
        return self._paths
    
    @classmethod
    def from_folder(cls, folder, raise_error=True, load_label_csv=True):
        """Creates a new instance of this class from the given `docdict`.
        
        Parameters
        ----------
        folder : string
            full path to folder name
        dbinterface : vedb-store.docdb.dbinterface
            db interface
        raise_error : bool
            Whether to raise an error if fields are missing. True simply raises an error, 
            False allows manual input of missing fields. 



        """
        ob = cls.__new__(cls)
        if not isinstance(folder, pathlib.Path):
            folder = pathlib.Path(folder)
        # Check if folder is locally defined
        if folder.parent == pathlib.Path('.'):
            if folder.exists():
                folder = folder.absolute()
            else:
                folder = pathlib.Path(BASE_PATH) / folder.name
        if not folder.exists():
            raise ValueError(f"Folder {folder.name} not found!")
        # Check on files available in folder
        missing_files = _check_paths(folder)
        if (len(missing_files) > 0) & raise_error:
            raise ValueError(f'Missing files: {missing_files}\n')

        # VEDB specific things: folder name as date, csv file for task, location labels
        # look for session csv for vedb
            
        # Set date    (& be explicit about what constitutes date)    
        try:
            # Assume year_month_day_hour_min_second for date specification in folder title
            session_date = datetime.datetime.strptime(folder.name, '%Y_%m_%d_%H_%M_%S')
            date = folder.name
        except:
            # Switch on verbosity?
            date = None
            print('Folder name not parseable as a date')

        ob.__init__(
              folder=folder.name,
              )
        
        return ob


@functools.total_ordering
class SessionClip(object):
    """Clip of a session: the time interval [onset, offset) on the session's clock.

    Clips compare by value: two clips are equal if session, onset, offset and
    tag are equal, and clips sort by (session, onset, offset). They are
    hashable on the same fields, so they can be used in sets / as dict keys;
    note that `dilate(overwrite=True)` changes a clip in place (and so its
    hash), so don't do that to clips stored in a set or dict.

    The tag is by convention 'location:task'; tags of merged clips are
    combined with '|' (e.g. 'kitchen:cooking|kitchen:cleaning'). See
    `location` and `task`.
    """
    def __init__(self, onset, offset, session=None, tag=None):
        """Representation of a time inteval (event, clip) within a session in VEDB
        
        Times used with this class should be on a clock with zero at the first frame
        of the world video.
        
        Parameters
        ----------
        onset : scalar 
            Time for start of clip. Times are given w.r.t. common 
            zero of all timestreams in VEDB data.
        offset : scalar 
            Time for end of clip. Times are given w.r.t. common 
            zero of all timestreams in VEDB data.
        session : str
            string name (date) for session to which this clip belongs
        tag : str
            descriptive string for this clip. By convention, often
            '<location>:<task>'

        """
        self.onset = onset
        self.offset = offset
        self.session = session
        self.tag = tag
        # lazy loading
        self.gaze_paths = None
        self._session_obj = None
        self._pupil = None
        self._gaze = None
        self._error = None
        self._odometry = None
    
    @classmethod
    def from_indices(cls, onset_i, offset_i, timestamps, session=None, tag=None):
        ob = cls.__new__(cls)
        ob.__init__(timestamps[onset_i], timestamps[offset_i], session=session, tag=tag)
        return ob
    
    @property
    def duration(self,):
        return self.offset - self.onset
    
    def _tag_parts(self):
        """(location, task) for each '|'-separated part of the tag; a part
        without ':' is a task with location None"""
        if self.tag is None:
            return []
        parts = []
        for part in str(self.tag).split('|'):
            if ':' in part:
                location, task = part.split(':', 1)
            else:
                location, task = None, part
            parts.append((location, task))
        return parts

    @property
    def task(self,):
        """Task(s) from the tag ('location:task'); unique tasks of a combined
        tag are joined with '|'. None if there is no task."""
        return _join_unique([task for _, task in self._tag_parts()])
    
    @property
    def location(self,):
        """Location(s) from the tag ('location:task'); unique locations of a
        combined tag are joined with '|'. None if there is no location."""
        return _join_unique([location for location, _ in self._tag_parts()])

    def _value(self):
        return (self.session, float(self.onset), float(self.offset), self.tag)

    def _sort_key(self):
        # Clips without a session sort last
        return (self.session is None, '' if self.session is None else str(self.session),
                float(self.onset), float(self.offset))

    def __eq__(self, other):
        if not isinstance(other, SessionClip):
            return NotImplemented
        return self._value() == other._value()

    def __lt__(self, other):
        if not isinstance(other, SessionClip):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __hash__(self):
        return hash(self._value())
    
    def load_gaze_pipeline(self, pipeline=None, clock='native', eye=None, validation_epoch=0):
        """Load specific gaze pipeline. Allows for non-default gaze to be loaded.

        Sets `self.gaze_paths`; see `Session.load_gaze_pipeline` for arguments.
        `pipeline` keys follow `vedb_gaze.utils.make_file_strings`, and tags
        not given are taken from `pipeline_default`.
        """
        self.clock = clock
        if self.session is None:
            return None
        self.gaze_paths = make_file_strings(**_get_pipeline(pipeline), **_session_file_kwargs(self.session),
                                            eye=eye, validation_epoch=validation_epoch)

        #self._error = dict((lr, np.load(self.gaze_paths['gaze']%lr)) for lr in eye)

    def quickshow(self, n=5, buffer=0.05, with_gaze_box=True, axs=None):
        if self.session is None:
            raise ValueError('Must have session defined to work.')
        ar = 4/3 # Questionable to hard code this
        rect_size = (600, 600)
        ses = Session(self.session)
        st, fin = self.indices(ses.world_time)
        dur = fin - st
        # Allow a buffer at start and end
        st = int(np.round(st + dur * buffer))
        fin = int(np.round(fin - dur * buffer))
        ii = np.round(np.linspace(st, fin, n)).astype(int)
        samples = []
        for i in ii:
            # TO DO: incorporate option for gaze-centered
            img = ses.load('world_camera', size=0.25, frame_idx=(i, i+1))[1][0]
            samples.append(img)
        if axs is None:
            scale = 2
            fig, axs = plt.subplots(1, n, figsize=(n * scale * ar, scale))
            do_tight_layout = True
        else:
            do_tight_layout = False
        if with_gaze_box:
            gaze = utils.match_time_points(dict(timestamp=ses.world_time), self.gaze_best)
            _, vh, vw, _ = file_io.list_array_shapes(str(ses.paths['world_camera'][1]))
            rect_width = rect_size[0] / vw
            rect_height = rect_size[1] / vh            
        for j, img, ax in zip(ii, samples, axs.flatten()):
            ax.imshow(img, extent=(0, 1, 1, 0), aspect='auto')
            ax.axis('off')
            # Show gaze rect
            if with_gaze_box:
                gaze_rect(gaze['norm_pos'][j], rect_width, rect_height, ax=ax, linewidth=2, edgecolor=(1, 0.95, 0))
        if do_tight_layout:
            plt.tight_layout()

    @property
    def pupil(self):
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._pupil is None:
            self._pupil = dict((lr, self(dict(np.load(PROC_PATH / self.session / (self.gaze_paths['pupil']%lr),
                                                      allow_pickle=True)))) \
                               for lr in ['left','right'])
            # Enumerate options here for clock
            if self.clock != 'native':
                if self.clock == 'world_time':
                    this_time = self.load('world_time')
                else:
                    raise NotImplementedError("can only handle 'native' and 'world_time' for clocks so far")
                for e in self._pupil.keys():
                    self._pupil[e] = utils.match_time_points(dict(timestamp=this_time), self._pupil[e])
        return self._pupil
    
    @property
    def gaze(self):
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._gaze is None:
            self._gaze = dict((lr, self(dict(np.load(PROC_PATH / self.session / (self.gaze_paths['gaze']%lr))))) \
                               for lr in ['left','right'])
            # Enumerate options here for clock
            if self.clock != 'native':
                if self.clock == 'world_time':
                    this_time = self.load('world_time')
                else:
                    raise NotImplementedError("can only handle 'native' and 'world_time' for clocks so far")
                #this_time = getattr(self, this_time)
                for e in self._gaze.keys():
                    self._gaze[e] = utils.match_time_points(dict(timestamp=this_time), self._gaze[e])
        return self._gaze

    @property
    def error(self):
        """Gaze error for both eyes (whole session, not clipped), for the
        validation epoch set in `load_gaze_pipeline()` (epoch 0 by default)"""
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._error is None:
            # make_file_strings returns a list of error files, one per
            # validation epoch; load_gaze_pipeline asks for a single epoch
            self._error = dict((lr, dict(np.load(PROC_PATH / self.session / (self.gaze_paths['error'][0]%lr)))) \
                               for lr in ['left','right'])
        return self._error

    @property
    def gaze_best(self):
        """Gaze for the eye with lower weighted gaze error (as in
        `Session.load_gaze(eye='best')`)"""
        if self.error['left']['gaze_err_weighted'] < self.error['right']['gaze_err_weighted']:
            return self.gaze['left']
        else:
            return self.gaze['right']

    def binary(self, timestamps, comparison_type=('>=', '<'), pre=0, post=0):
        """Get binary index for this clip within `timestamps`
        
        Parameters
        ----------
        timestamps : array-like
            array of timestamps. Method returns a vector of True values for all
            timestamps within this array that are after self.onset and before 
            self.offset (see next kwarg for nuance of how these are handled)
        comparison_type: tuple
            two-tuple for how to handle boundaries of clip. First value in the 
            tuple is '>' or '>=', second value is '<' or '<='. These strings 
            indicate whether to take all times e.g. GREATER THAN the onset or
            GREATER THAN OR EQUAL TO the onset. 
        pre : scalar
            time in seconds to extend onset back in time (PRIOR to onset)
        post : scalar
            time in seconds to extend offset forward in time (AFTER onset)
        """
        if comparison_type[0] == '>=':
            after_start = timestamps >= (self.onset - pre)
        elif comparison_type[0] == '>':
            after_start = timestamps > (self.onset - pre)
        else:
            raise ValueError('only "<=", "<", ">=", or ">" are allowed for elements of `comparison_type`')
        if comparison_type[1] == '<=':
            before_end = timestamps <= (self.offset + post)
        elif comparison_type[1] == '<':
            before_end = timestamps < (self.offset + post)
        else:
            raise ValueError('only "<=", "<", ">=", or ">" are allowed for elements of `comparison_type`')
        return after_start & before_end

    def dilate(self, pre=0, post=0, tlimits=(0, np.inf), overwrite=False):
        """Dilate this clip in time by adding `pre` to onset and `post` 
        to offset.
        """
        on = np.maximum(tlimits[0], self.onset - pre)
        off = np.minimum(tlimits[1], self.offset + post)
        if overwrite:
            self.onset = on
            self.offset = off
        else:
            return SessionClip(on, off, self.session, tag=self.tag)

    def times(self, include_duration=False):
        if include_duration:
            return self.onset, self.offset, self.duration
        else:
            return self.onset, self.offset    
        
    def indices(self, timestamps, comparison_type=('>=', '<')):
        binary_index = self.binary(timestamps, comparison_type=comparison_type)
        if ~np.any(binary_index):
            # Timesamps provided do not contain any within bounds of this clip
            # Throw error?
            return None, None
        integer_indices, = np.nonzero(binary_index)
        return (integer_indices[0], integer_indices[-1]+1)
    
    def __call__(self, inpt, key=None, pre=0, post=0):
        """Select the samples of an arraydict that fall within this clip

        Parameters
        ----------
        inpt : dict
            arraydict with a 'timestamp' field (on the session clock)
        key : str, optional
            if given, return only this field
        pre, post : scalar, optional
            seconds to extend the clip before / after, by default 0

        Returns
        -------
        dict or array
            fields with one value per timestamp, restricted to the clip;
            other fields (dicts, scalars, metadata) are left out
        """
        assert isinstance(inpt, dict), "Input must be dictionary!"
        assert 'timestamp' in inpt, "dict input must have 'timestamp' field!"
        tt = np.asarray(inpt['timestamp'])
        output = _clip_arraydict(dict(inpt, timestamp=tt), self.binary(tt, pre=pre, post=post))
        if key is not None:
            return output[key]
        return output
        
    def load(self, data_type, processed=True, **kwargs):
        """Load data for this clip

        Parameters
        ----------
        data_type : str
            'world_camera', 'world_time', 'gaze', 'odometry', 'eye_left' or
            'eye_right'
        processed : bool, optional
            odometry only: if True (default), use
            `vedb_gaze.odometry.load_odometry` (resampled, smoothed, with
            'pitch', 'roll', 'yaw'); if False, the raw odometry.pldata
            records, as a dict of arrays
        **kwargs
            passed to the loading function (for processed odometry, to
            `load_odometry`)
        """
        if self.session is None:
            raise ValueError('Cannot load data with clip `session` set')
        if data_type == 'world_camera':
            fname = BASE_PATH / self.session / 'worldPrivate.mp4'
            ftime = BASE_PATH / self.session / 'world_timestamps_0start.npy'
            world_time = np.load(ftime)
            data = file_io.load_video(fname, frames=self.indices(world_time), **kwargs)
            time = world_time[self.binary(world_time)]
            return time, data
        elif data_type == 'world_time': 
            ftime = BASE_PATH / self.session / 'world_timestamps_0start.npy'
            world_time = np.load(ftime)
            time = world_time[self.binary(world_time)]
            return time
        elif data_type == 'gaze':
            ses = Session(self.session)
            gaze = ses.load_gaze(**kwargs)
            return self(gaze)
        elif (data_type == 'odometry') and processed:
            odo = load_odometry(BASE_PATH / self.session, **kwargs)
            return _clip_arraydict(odo, self.binary(odo['timestamp']))
        elif data_type == 'odometry':
            odo = file_io.load_msgpack((BASE_PATH / self.session / 'odometry.pldata'))
            odo = utils.dictlist_to_arraydict(odo)
            # Replace odometry time with 0-referenced time
            odo_t = np.load(BASE_PATH / self.session / 'odometry_timestamps_0start.npy')
            odo['timestamp'] = odo_t
            return self(odo)
        elif data_type in ('eye_left', 'eye_right'):
            lr = '1' if 'left' in data_type else '0'
            fname = BASE_PATH / self.session / f'eye{lr}{EYE_PRIVACY_SETTING}.mp4'
            ftime = BASE_PATH / self.session / f'eye{lr}_timestamps_0start.npy'
            data_time = np.load(ftime)
            data = file_io.load_video(fname, frames=self.indices(data_time), **kwargs)
            time = data_time[self.binary(data_time)]            
            return time, data
        else:
            raise NotImplementedError('Coming soon!')
            
    def sample(self,
           n_samples=1,
           sample_duration=3,
           start_jitter_fraction=0.5, # This seems to create a bias toward the center over iterations, which is... fine?
          frac_to_vary=0.2):
        """Choose `n_samples` non-overlapping random samples within this clip

        Raises
        ------
        ValueError
            if the clip is shorter than `n_samples * sample_duration`
        """
        if self.duration < n_samples * sample_duration:
            raise ValueError(f'Clip is {self.duration:.2f} s long, too short for {n_samples} '
                             f'samples of {sample_duration} s')
        wiggle_room = self.duration - (sample_duration * n_samples)
        start_jitter = wiggle_room * np.random.rand() * start_jitter_fraction
        wiggle_room -= start_jitter
        start_times = np.arange(0, n_samples*sample_duration, sample_duration).astype(float)
        start_times += start_jitter
        time_to_add = get_time_split(wiggle_room * np.random.rand(), n_samples, frac_to_vary=frac_to_vary)
        start_times += np.cumsum(time_to_add)
        start_times += self.onset
        output = np.array([start_times, start_times+sample_duration]).T
        return [SessionClip(*times, tag=self.tag, session=self.session) for times in output]

    def shade_bg(self, ax=None, fcol=(.9, .9, .9), yl=None, vert=False, zorder=-1):
        """Shade in xtick grid (every other tick mark is gray/white)"""
        if ax is None:
            fig, ax = plt.subplots()
        st, fin = self.onset, self.offset
        if yl is None:
            yl = plt.ylim()
        if vert:
            # interpret yl as xlim, w as height
            ax.fill([yl[0], yl[0], yl[1], yl[1]],
                [st, fin, fin, st],
                color=fcol, edgecolor='none', zorder=zorder)
            plt.xlim(yl)
        else:
            ax.fill([st, fin, fin, st],
                [yl[0], yl[0], yl[1], yl[1]],
                color=fcol, edgecolor='none', zorder=zorder)
            plt.ylim(yl)
        

    def make_gaze_animation(self,
                            rect_size=(600,600),
                            pupil_str=None,
                            gaze_str=None,
                            fps=30, 
                            world_size_factor=0.25,
                            hspace=0.1,
                            wspace=None,
                            eye_left_color=(1.0, 0.5, 0.0),  # orange
                            eye_right_color=(0.0, 0.5, 1.0),  # cyan
                            raise_error=False,
                        ):
        """Make radical gaze animation

        Parameters
        ----------
        pupil_str, gaze_str : str or None, optional
            pupil / gaze file name templates (with '%s' for the eye) in
            PROC_PATH / <session>. By default None, which uses this clip's
            pipeline (`self.gaze_paths`, set by `load_gaze_pipeline()`; the
            vedb_gaze default tags if it has not been called).
        """
        if (pupil_str is None) or (gaze_str is None):
            if self.gaze_paths is None:
                self.load_gaze_pipeline()
            if pupil_str is None:
                pupil_str = self.gaze_paths['pupil']
            if gaze_str is None:
                gaze_str = self.gaze_paths['gaze']
        if ('%s' not in pupil_str) or ('%s' not in gaze_str):
            raise ValueError("pupil / gaze file names need a '%s' placeholder for the eye; "
                             "call load_gaze_pipeline() with eye=None")
        global eye_left_frame
        global eye_left_image
        global eye_right_frame    
        global eye_right_image
        eye_video_size = 400 # x 400, square
        si = get_session_info(self.session)
        
        try:
            ses = Session.from_folder(self.session, raise_error=raise_error)
            pl = PROC_PATH / ses.folder / (pupil_str%'left')
            gl = PROC_PATH / ses.folder / (gaze_str%'left')
            pr = PROC_PATH / ses.folder / (pupil_str%'right')
            gr = PROC_PATH / ses.folder / (gaze_str%'right')
            include_left_eye = 'eye_left' in ses.paths
            include_right_eye = 'eye_right' in ses.paths
            
            if include_left_eye:
                eltf, elvf = ses.paths['eye_left']
                eye_left_frame = 0
                eye_left_time = np.load(eltf)
                eye_left_vid = ses.get_video_handle('eye_left')
            if include_right_eye:
                ertf, ervf = ses.paths['eye_right']    
                eye_right_frame = 0
                eye_right_time = np.load(ertf)
                eye_right_vid = ses.get_video_handle('eye_right')
            
            _, vh, vw, _ = file_io.list_array_shapes(str(ses.paths['world_camera'][1]))
            n_frames = len(self(dict(timestamp=ses.world_time))['timestamp'])
            #frame = world[0]
            rect_width = rect_size[0] / vw
            rect_height = rect_size[1] / vh
            ar = vw / vh

            fig = plt.figure(figsize=(8, 8 * 13.5/12)) #* 14 / 12))
            gs = GridSpec(2,3, figure=fig,  hspace=hspace, wspace=wspace,
                        height_ratios=[1, 2], width_ratios=[1, 1, 1])
            ax_eye_left = fig.add_subplot(gs[0,0])
            ax_eye_right = fig.add_subplot(gs[0,1])
            ax_gc = fig.add_subplot(gs[0, 2])
            ax_vid = fig.add_subplot(gs[1,:])
            ax_vid.axis([0, 1, 1, 0])
            ax_vid.set_xticks([]) # To visual field size?
            ax_vid.set_yticks([])

            # Initialize all plots
            if gl.exists():
                gaze_left = dict(np.load(gl))
                gaze_rect_pixel_size = rect_size
                # FIX ME
                wt = ses.world_time[self.binary(ses.world_time)]
                g_matched = utils.match_time_points(dict(timestamp=wt), gaze_left)
                _, gaze_centered_video = wt, wv = self.load('world_camera', 
                                                            center=g_matched['norm_pos'],
                                                            crop_size=gaze_rect_pixel_size)

            if gr.exists():
                gaze_right = dict(np.load(gr))
            world_time, world = self.load('world_camera', size=world_size_factor)
            world_h = ax_vid.imshow(world[0], extent=[0, 1, 1, 0], aspect='auto')
            # For now: choose best, don't rely on left.
            if gl.exists():
                gaze_rect_lh = gaze_rect(gaze_left['norm_pos'][0], rect_width, rect_height, ax=ax_vid, linewidth=3, edgecolor=eye_left_color)
                gaze_dot_lh = ax_vid.scatter(*gaze_left['norm_pos'][0], c=eye_left_color)
                gc_h = ax_gc.imshow(gaze_centered_video[0], extent=[0, 1, 1, 0])
            else:
                gaze_rect_lh = gaze_rect([-1,-1], rect_width, rect_height, ax=ax_vid, linewidth=3)
                gaze_dot_lh = ax_vid.scatter(*[-1,-1], c='black')
                gc_h = ax_gc.imshow(np.zeros((200,200)), extent=[0, 1, 1, 0])

            if gr.exists():
                gaze_rect_rh = gaze_rect(gaze_right['norm_pos'][0], rect_width, rect_height, ax=ax_vid, linewidth=3, edgecolor=eye_right_color)
                gaze_dot_rh = ax_vid.scatter(*gaze_right['norm_pos'][0], c=eye_right_color)
                #gc_h = ax_gc.imshow(gaze_centered_video[0], extent=[0, 1, 1, 0])
            else:
                gaze_rect_rh = gaze_rect([-1,-1], rect_width, rect_height, ax=ax_vid, linewidth=3)
                gaze_dot_rh = ax_vid.scatter(*[-1,-1], c='black')
                #gc_h = ax_gc.imshow(np.zeros((200,200)), extent=[0, 1, 1, 0])

                
            if include_left_eye:
                success, eye_left_image = eye_left_vid.VideoObj.read()
                eye_left_h = ax_eye_left.imshow(eye_left_image, extent=[0, 1, 1, 0])
                
            if pl.exists():
                pupil_left = dict(np.load(pl, allow_pickle=True))
                ellipse_data_left = dict((k, np.array(v) / eye_video_size)
                                        for k, v in pupil_left['ellipse'][0].items())
                pupil_left_eh, pupil_left_dh = show_ellipse(ellipse_data_left,
                                        center_color=eye_left_color,
                                        facecolor=eye_left_color +
                                        (0.5,),
                                        ax=ax_eye_left)

            if include_right_eye:
                success, eye_right_image = eye_right_vid.VideoObj.read()
                eye_right_h = ax_eye_right.imshow(eye_right_image, extent=[0, 1, 1, 0])
                
            if pr.exists():
                pupil_right = dict(np.load(pr, allow_pickle=True))
                ellipse_data_right = dict((k, np.array(v) / eye_video_size)
                                        for k, v in pupil_right['ellipse'][0].items())
                pupil_right_eh, pupil_right_dh = show_ellipse(ellipse_data_right,
                                        center_color=eye_right_color,
                                        facecolor=eye_right_color +
                                        (0.5,),
                                        ax=ax_eye_right)
                
            ax_eye_left.axis([1, 0, 1, 0]) # [0,1, 0, 1]
            #ax_eye_left.axis([0, 1, 0, 1]) # [0,1, 0, 1]
            ax_eye_left.set_xticks([])
            ax_eye_left.set_yticks([])
            
            ax_eye_right.axis([1, 0, 1, 0]) # [0,1, 0, 1]
            #ax_eye_right.axis([0, 1, 0, 1]) #[1, 0, 1, 0]
            ax_eye_right.set_xticks([])
            ax_eye_right.set_yticks([])

            
            gaze_box_vis_degrees = si['fov'] * rect_width
            vmx_deg = gaze_box_vis_degrees / 2
            tick_labels = ['%.1f'%x for x in [-vmx_deg, 0, vmx_deg]]
            ax_gc.set_xticks([0, 0.5, 1])
            ax_gc.set_xticklabels(tick_labels)
            ax_gc.set_yticks([0, 0.5, 1])
            ax_gc.set_yticklabels(tick_labels)
            ax_gc.grid('on', linestyle=':', color=(0.95, 0.85, 0))
            #plt.close(fig)
            
            def set_ellipse(ellipse_h, dot_h, ellipse, frame):
                """h_s are two handles: for ellipse, for center dot"""
                tmp = ellipse[frame]
                ellipse_data = dict((k, np.array(v) / eye_video_size)
                            for k, v in tmp.items())
                ellipse_h.set_center(ellipse_data['center'])
                ellipse_h.set_height(ellipse_data['axes'][1])
                ellipse_h.set_width(ellipse_data['axes'][0])
                ellipse_h.set_angle(tmp['angle'])
                # Accumulate?
                dot_h.set_offsets([ellipse_data['center']])
                return ellipse_h, dot_h
            
            def init():
                to_return = [world_h]
                world_h.set_array(np.zeros_like(world[0]))
                
                if gl.exists():
                    gaze_rect_lh.set_xy([0.5, 0.5] - np.array([rect_width/2, rect_height/2]))
                    gaze_dot_lh.set_offsets([0.5, 0.5])
                    gc_h.set_array(np.zeros_like(gaze_centered_video[0]))
                    to_return.extend([gaze_rect_lh, gaze_dot_lh, gc_h])
                
                if include_left_eye:
                    eye_left_h.set_array(np.zeros((400,400), dtype=np.uint8))
                    to_return.append(eye_left_h)
                    
                if pl.exists():
                    _ = set_ellipse(pupil_left_eh, pupil_left_dh,
                                                                pupil_left['ellipse'], 0)
                    to_return.extend([pupil_left_eh, pupil_left_dh])
                    
                if include_right_eye:
                    eye_right_h.set_array(np.zeros((400,400), dtype=np.uint8)) #np.zeros_like(eye_right_ds[0]))
                    to_return.append(eye_right_h)
                    
                if pr.exists():
                    _ = set_ellipse(pupil_right_eh, pupil_right_dh,
                                                                pupil_right['ellipse'], 0)
                    to_return.extend([pupil_right_eh, pupil_right_dh])
                
                return to_return

            def animate(i):
                global eye_left_frame
                global eye_right_frame
                global eye_right_image
                global eye_left_image
                #success, world_im = world_vid.VideoObj.read()
                world_h.set_data(world[i]) # world_im)
                world_time_this_frame = world_time[i]
                to_return = [world_h]
                
                
                if include_left_eye:
                    while eye_left_time[eye_left_frame] < world_time_this_frame:
                        eye_left_frame += 1
                        success, eye_left_image = eye_left_vid.VideoObj.read()
                    eye_left_h.set_data(eye_left_image)
                    to_return.append(eye_left_h)
                if pl.exists():
                    _ = set_ellipse(pupil_left_eh, pupil_left_dh,
                                                                pupil_left['ellipse'], eye_left_frame)
                    to_return.extend([pupil_left_eh, pupil_left_dh])
                if include_right_eye:
                    while eye_right_time[eye_right_frame] < world_time_this_frame:
                        eye_right_frame += 1
                        success, eye_right_image = eye_right_vid.VideoObj.read()
                    eye_right_h.set_data(eye_right_image)
                    to_return.append(eye_right_h)
                if pr.exists():
                    _ = set_ellipse(pupil_right_eh, pupil_right_dh,
                                                                pupil_right['ellipse'], eye_right_frame)
                    to_return.extend([pupil_right_eh, pupil_right_dh])

                if gl.exists():
                    gaze_rect_lh.set_xy(gaze_left['norm_pos'][eye_left_frame] - np.array([rect_width/2, rect_height/2]))
                    gaze_dot_lh.set_offsets(gaze_left['norm_pos'][eye_left_frame])
                    try:
                        gc_h.set_data(gaze_centered_video[i])
                    except:
                        pass
                    to_return.extend([gaze_rect_lh, gaze_dot_lh, gc_h])
                    
                if gr.exists():
                    gaze_rect_rh.set_xy(gaze_right['norm_pos'][eye_right_frame] - np.array([rect_width/2, rect_height/2]))
                    gaze_dot_rh.set_offsets(gaze_right['norm_pos'][eye_right_frame])
                    to_return.extend([gaze_rect_rh, gaze_dot_rh])

                
                return to_return



            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1/fps * 1000, blit=True)
        except:
            try:
                eye_left_vid.VideoObj.release()
                eye_right_vid.VideoObj.release()
            except:
                pass
            raise
        return anim

        
    def __repr__(self,):
        return (f'Clip for {self.session}\n'
                f'> {self.duration:.2f}s, from {self.onset:.2f}-{self.offset:.2f}\n'
                f'> {self.tag}'
               )
    

def _join_unique(values):
    """Unique non-empty values (in order) joined with '|'; None if there are none"""
    out = []
    for v in values:
        if (v is None) or (v == ''):
            continue
        if v not in out:
            out.append(v)
    return '|'.join(out) if len(out) > 0 else None


def _combine_tags(clips):
    """Tag for a clip made from `clips`: their distinct tags (or tag parts of
    already-combined tags) in time order, joined with '|'; None if none"""
    parts = []
    for clip in _sort_by_time(clips):
        if clip.tag is not None:
            parts.extend(str(clip.tag).split('|'))
    return _join_unique(parts)


def _sort_by_time(clips):
    """Clips sorted by (onset, offset, tag), deterministic for equal times"""
    return sorted(clips, key=lambda c: (float(c.onset), float(c.offset), str(c.tag)))


def _first_session(clips):
    """First session that is not None among `clips` (None if all are None)"""
    for clip in clips:
        if clip.session is not None:
            return clip.session
    return None


def _merge_clips(clips):
    """Merge clips that overlap or touch (one session) into single clips

    The merged clip spans all the clips it is made from; its tag combines
    their tags (see `_combine_tags`).
    """
    groups = []
    for clip in _sort_by_time(clips):
        if (len(groups) > 0) and (clip.onset <= groups[-1][1]):
            groups[-1][1] = max(groups[-1][1], clip.offset)
            groups[-1][2].append(clip)
        else:
            groups.append([clip.onset, clip.offset, [clip]])
    return [SessionClip(on, off, session=_first_session(cc), tag=_combine_tags(cc))
            for on, off, cc in groups]


def _intersect_clips(clips_a, clips_b):
    """Time shared by `clips_a` and `clips_b` (one session), merged

    Each piece of shared time gets the combined tags of the two clips it
    comes from. Clips that only touch share no time.
    """
    pieces = []
    for a in clips_a:
        for b in clips_b:
            on, off = max(a.onset, b.onset), min(a.offset, b.offset)
            if off > on:
                pieces.append(SessionClip(on, off, session=_first_session([a, b]),
                                          tag=_combine_tags([a, b])))
    return _merge_clips(pieces)


def _subtract_clips(clips_a, clips_b):
    """Time of `clips_a` not covered by `clips_b` (one session)

    Clips of `clips_a` are trimmed or split; the pieces keep the tag of the
    clip they come from (pieces are not merged with each other).
    """
    out = []
    for a in clips_a:
        pieces = [(a.onset, a.offset)]
        for b in clips_b:
            remaining = []
            for on, off in pieces:
                if (b.offset <= on) or (b.onset >= off):
                    # No shared time
                    remaining.append((on, off))
                    continue
                if b.onset > on:
                    remaining.append((on, b.onset))
                if b.offset < off:
                    remaining.append((b.offset, off))
            pieces = remaining
        out.extend(SessionClip(on, off, session=a.session, tag=a.tag) for on, off in pieces if off > on)
    return _sort_by_time(out)


def _common_clips(clips_a, clips_b):
    """Whole clips of either list that overlap (or touch) a clip of the other
    list, with overlapping / touching clips merged (one session; symmetric)"""
    def _touches(clip, others):
        return any((o.onset <= clip.offset) and (o.offset >= clip.onset) for o in others)
    keep = [a for a in clips_a if _touches(a, clips_b)] + [b for b in clips_b if _touches(b, clips_a)]
    return _merge_clips(keep)


def _group_by_session(clips):
    """dict of session -> list of clips"""
    out = {}
    for clip in clips:
        out.setdefault(clip.session, []).append(clip)
    return out


def _timeline_end(timestamps):
    """Time just after the last sample: last timestamp + median sample interval"""
    timestamps = np.asarray(timestamps)
    if len(timestamps) < 2:
        return timestamps[-1]
    return timestamps[-1] + np.median(np.diff(timestamps))


def _indices_to_times(onoff_indices, native_timestamps):
    """(onset, offset) sample indices -> times in `native_timestamps`

    Offsets are exclusive (index of the first sample after the clip). An
    offset of len(native_timestamps) (a clip still on at the end, as given by
    `onoff_from_binary`) is mapped to one sample interval past the last
    timestamp, so that the clip includes the last sample.
    """
    native_timestamps = np.asarray(native_timestamps)
    n = len(native_timestamps)
    out = []
    for ii in onoff_indices:
        on_i, off_i = int(ii[0]), int(ii[1])
        on = native_timestamps[on_i]
        off = native_timestamps[off_i] if off_i < n else _timeline_end(native_timestamps)
        out.append((on, off))
    return out


class SessionClipList(object):
    """Clips within ONE session, on that session's timeline

    Logical operators work on time (on the session clock):

    - `A + B`: union; clips that overlap or touch are merged into one.
    - `A & B`: time shared by A and B (merged); touching clips share no time.
    - `A - B`: time of A not covered by B; clips of A are trimmed or split.
    - `A.common_clips(B)`: whole clips of A or B that overlap (or touch) a
      clip of the other list, merged; same result as `B.common_clips(A)`.

    Tags: trimmed / split clips keep their tag; merged clips and pieces of
    shared time get the distinct tags combined with '|' (e.g. 'a:b|c:d').
    Clips made from a binary mask (`from_binary`, `invert`) have tag None.
    Operators between lists of different sessions raise ValueError (use
    `ClipList` for clips from several sessions); a list with session None
    combines with any session. Empty results are empty SessionClipLists.
    """
    def __init__(self, onsets_offsets, native_timestamps, session=None, tags=None):
        """List of clips
        
        all times passed to this and its methods must be normalized to session start
        i.e. starting when the first data stream for a session came online.

        Parameters
        ----------
        onsets_offsets : array-like
            (n, 2) onset, offset times (a third column, e.g. duration, is ignored)
        native_timestamps : array
            timestamps of the session timeline (e.g. world video time)
        session : str, optional
            session (folder) name
        tags : list, optional
            one tag per clip
        """
        self.clip_list = []
        self.native_timestamps = native_timestamps
        self.session = session
        if not isinstance(onsets_offsets, np.ndarray):
            onsets_offsets = np.asarray(onsets_offsets)
        if onsets_offsets.size == 0:
            # No clips
            onsets_offsets = onsets_offsets.reshape(0, 2)
        for j, onoff in enumerate(onsets_offsets):
            onset, offset = onoff[:2]
            if tags is None:
                tag = None
            else:
                tag = tags[j]
            self.clip_list.append(SessionClip(onset, offset, session=session, tag=tag))

    def _new(self, clips, session=None):
        """SessionClipList with `clips` on the same timeline"""
        if session is None:
            session = self.session
        return SessionClipList([(c.onset, c.offset) for c in clips], self.native_timestamps,
                               session=session, tags=[c.tag for c in clips])

    def binary(self, timestamps=None, comparison_type=('>=', '<')):
        """Return binary version of this list over full duration of 
        `native_timestamps` (or of `timestamps`, if given)
        
        i.e., convert this SessionClipList to a vector of True values for native 
        timestamps *during* clips and False values for native timestamps 
        *outside* of clips. For extended clips, `dilate()` first.
        """
        if timestamps is None:
            timestamps = self.native_timestamps
        timestamps = np.asarray(timestamps)
        out = np.zeros(timestamps.shape, dtype=bool)
        for clip in self:
            out |= clip.binary(timestamps, comparison_type=comparison_type)
        return out

    def indices(self, timestamps=None, comparison_type=('>=', '<')):
        """(n_clips, 2) array of (start, end) indices of each clip in
        `timestamps` (default `native_timestamps`); end is exclusive. A clip
        with no samples gives start == end."""
        if timestamps is None:
            timestamps = self.native_timestamps
        timestamps = np.asarray(timestamps)
        side_on = 'left' if comparison_type[0] == '>=' else 'right'
        side_off = 'left' if comparison_type[1] == '<' else 'right'
        out = [(np.searchsorted(timestamps, clip.onset, side=side_on),
                np.searchsorted(timestamps, clip.offset, side=side_off)) for clip in self]
        out = np.array(out, dtype=int).reshape(-1, 2)
        out[:, 1] = np.maximum(out[:, 0], out[:, 1])
        return out

    @property
    def native_indices(self):
        """(n_clips, 2) array of (start, end) indices of each clip in `native_timestamps`"""
        return self.indices()
    
    def times(self, include_duration=False):
        return np.asarray([clip.times(include_duration) for clip in self.clip_list])
        
    def filter_duration(self, min_duration=0, max_duration=np.inf):
        """Clips with min_duration < duration < max_duration (seconds)"""
        return self._new([x for x in self if (x.duration > min_duration) and (x.duration < max_duration)])

    def dilate(self, pre=0, post=0, merge=False):
        """dilate time for each clip. Times in seconds.

        Clips are limited to the timeline (`native_timestamps`). If `merge`,
        dilated clips that overlap or touch are merged (tags combined).
        """
        # Don't overwrite self
        mn = np.min(self.native_timestamps)
        mx = _timeline_end(self.native_timestamps)
        clips = [x.dilate(pre=pre, post=post, tlimits=(mn, mx)) for x in self]
        if merge:
            clips = _merge_clips(clips)
        return self._new(clips)

    def invert(self, comparison_type=('>=', '<')):
        """Clips for the time between clips (on `native_timestamps`); tags are None"""
        out = SessionClipList.from_binary(~self.binary(comparison_type=comparison_type),
                                          self.native_timestamps, session=self.session)
        # Kill any 1-frame clips
        out.clip_list = [clip for clip in out if clip.duration > 0]
        return out

    @property
    def tags(self):
        return [x.tag for x in self]

    @classmethod
    def from_indices(cls, onoff_indices, native_timestamps, session=None, tags=None):
        ob = cls.__new__(cls)
        onoffs = _indices_to_times(onoff_indices, native_timestamps)
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob

    @classmethod
    def from_binary(cls, binary, native_timestamps, session=None, tags=None):
        """Clips for runs of True in `binary` (one value per native timestamp).
        Tags are None unless `tags` (one per run) is given."""
        ob = cls.__new__(cls)
        onoff_indices = onoff_from_binary(binary, return_duration=False)
        onoffs = _indices_to_times(onoff_indices, native_timestamps)
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob
    
    @classmethod
    def from_clips(cls, list_of_clips, native_timestamps):
        ob = cls.__new__(cls)
        # Require that all clips are from same session. For now, clip lists must tbe same session.
        session = _first_session(list_of_clips)
        assert all([x.session in (session, None) for x in list_of_clips]),\
              'SessionClip objects in SessionClipList must be from same session!'
        onoffs = [(x.onset, x.offset) for x in list_of_clips]
        tags = [x.tag for x in list_of_clips]
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob

    def to_cliplist(self, name='MyClipList'):
        """ClipList with the clips of this list"""
        return ClipList(list(self), name=name)

    def _check_other(self, other):
        """Session of the result of combining with `other`; ValueError if
        `other` is from a different session"""
        if not isinstance(other, SessionClipList):
            raise TypeError(f'Expected a SessionClipList, got {type(other).__name__}')
        if (self.session is not None) and (other.session is not None) and (self.session != other.session):
            raise ValueError(f"Can't combine clips of session {self.session} with clips of "
                             f"session {other.session}; use ClipList for clips from several sessions")
        return self.session if self.session is not None else other.session

    def __add__(self, cliplist):
        """Time union; overlapping or touching clips are merged (tags combined)"""
        session = self._check_other(cliplist)
        return self._new(_merge_clips(list(self) + list(cliplist)), session=session)

    def __and__(self, cliplist):
        """Time shared with `cliplist` (tags combined)"""
        session = self._check_other(cliplist)
        return self._new(_intersect_clips(list(self), list(cliplist)), session=session)

    def __sub__(self, cliplist):
        """Time of this list not covered by `cliplist` (clips trimmed / split)"""
        session = self._check_other(cliplist)
        return self._new(_subtract_clips(list(self), list(cliplist)), session=session)

    def common_clips(self, cliplist):
        """Whole clips of this list or `cliplist` that overlap (or touch) a
        clip of the other list, merged; `A.common_clips(B)` == `B.common_clips(A)`"""
        session = self._check_other(cliplist)
        return self._new(_common_clips(list(self), list(cliplist)), session=session)

    def __len__(self):
        return(len(self.clip_list))
    
    def __getitem__(self, i):
        return self.clip_list[i]
    
    def __iter__(self):
        return iter(self.clip_list)

    def __repr__(self):
        return f'SessionClipList for {self.session}: {len(self)} clips'


def _sort_unique(values):
    """Sorted unique values, None last"""
    return sorted(set(values), key=lambda v: (v is None, '' if v is None else str(v)))


class ClipList(object):
    """Clips from potentially many sessions (e.g. a sample of VEDB)

    Logical operators and `common_clips` work like those of SessionClipList
    (on time; see there), separately for each session: clips of different
    sessions never interact. Sessions in only one list pass through `+`,
    drop out of `&` and `common_clips`, and are unchanged by `-`. Results
    are sorted by (session, onset) and keep this list's name.
    """
    def __init__(self, clips, name='MyClipList', load_gaze_centered=False):
        """List of clips from potentially disparate sessions
        
        all times passed to this and its methods must be normalized to session start
        i.e. starting when the first data stream for a session came online.

        Parameters
        ----------
        clips : iterable
            SessionClips, or SessionClipLists / ClipLists (whose clips are
            added), or a mix; the list is copied
        """
        clip_list = []
        for clip in clips:
            if isinstance(clip, (SessionClipList, ClipList)):
                clip_list.extend(list(clip))
            else:
                clip_list.append(clip)
        self.clip_list = clip_list
        self.name = name
        self.load_gaze_centered = load_gaze_centered

    def _new(self, clips):
        return ClipList(sorted(clips), name=self.name, load_gaze_centered=self.load_gaze_centered)

    def dilate(self, pre=0, post=0, tlimits=(0, np.inf)):
        """Dilate time for each clip. 

        No notion of merging clips - all clips in a ClipList are treated as independent.
        SessionClipList clips are known to be on the same clock wrt each other, and can
        be merged.
        
        Parameters
        ----------
        pre : scalar
            time to add before clip in seconds.
        post : scalar
            time to add after clip in seconds.
        tlimits : tuple, optional
            (min, max) times to which dilated onsets / offsets are limited,
            by default (0, np.inf). A ClipList has no single session
            timeline, so offsets are not limited to each session's duration.
        """
        # Don't overwrite self
        clips = [x.dilate(pre=pre, post=post, tlimits=tlimits) for x in self]
        return ClipList(clips, name=self.name, load_gaze_centered=self.load_gaze_centered)

    @property
    def durations(self):
        return np.asarray([x.duration for x in self])

    @property
    def sessions(self):
        """Sorted unique sessions (None last)"""
        return _sort_unique(self.clip_sessions)

    @property
    def clip_sessions(self):
        """Session of each clip"""
        return [x.session for x in self]

    @property
    def session_count(self):
        """dict of session -> number of clips"""
        return unique(self.clip_sessions)[1]
    
    @property
    def tasks(self):
        """Sorted unique tasks (None last)"""
        return _sort_unique(self.clip_tasks)

    @property
    def clip_tasks(self):
        """Task of each clip"""
        return [x.task for x in self]

    @property
    def task_count(self):
        """dict of task -> number of clips"""
        return unique(self.clip_tasks)[1]

    @property
    def locations(self):
        """Sorted unique locations (None last)"""
        return _sort_unique(self.clip_locations)

    @property
    def clip_locations(self):
        """Location of each clip"""
        return [x.location for x in self]

    @property
    def location_count(self):
        """dict of location -> number of clips"""
        return unique(self.clip_locations)[1]
    
    @property
    def tags(self):
        """Tag of each clip"""
        return [x.tag for x in self]

    def times(self, include_duration=False):
            return np.asarray([clip.times(include_duration) for clip in self])

    def filter_duration(self, min_duration=0, max_duration=np.inf):
        """Durations in seconds"""
        output_clips = [x for x in self if (x.duration > min_duration) and (x.duration  < max_duration)]
        return ClipList(output_clips, name=self.name, load_gaze_centered=self.load_gaze_centered)

    def _per_session(self, cliplist, fn, left_only=False, right_only=False):
        """Apply `fn(clips_a, clips_b)` per session; sessions present only in
        this list (`left_only`) or only in `cliplist` (`right_only`) are
        passed through `fn` with an empty list for the other side if True,
        dropped if False"""
        a = _group_by_session(self)
        b = _group_by_session(cliplist)
        out = []
        for session in set(a) | set(b):
            clips_a, clips_b = a.get(session, []), b.get(session, [])
            if (len(clips_a) > 0) and (len(clips_b) > 0):
                out.extend(fn(clips_a, clips_b))
            elif (len(clips_a) > 0) and left_only:
                out.extend(fn(clips_a, []))
            elif (len(clips_b) > 0) and right_only:
                out.extend(fn([], clips_b))
        return self._new(out)

    def __add__(self, cliplist):
        """Time union per session; overlapping or touching clips are merged"""
        return self._per_session(cliplist, lambda a, b: _merge_clips(a + b), left_only=True, right_only=True)

    def __and__(self, cliplist):
        """Time shared with `cliplist`, per session"""
        return self._per_session(cliplist, _intersect_clips)

    def __sub__(self, cliplist):
        """Time not covered by `cliplist`, per session (clips trimmed / split)"""
        return self._per_session(cliplist, _subtract_clips, left_only=True)

    def common_clips(self, cliplist):
        """Whole clips of this list or `cliplist` that overlap (or touch) a
        clip of the other list in the same session, merged; symmetric"""
        return self._per_session(cliplist, _common_clips)

    def by_session(self, native_timestamps=None):
        """Split into one SessionClipList per session

        Parameters
        ----------
        native_timestamps : dict or array, optional
            timeline for each session's SessionClipList: a dict of session
            -> timestamps, or one array (only if there is one session). By
            default, each session's world video timestamps
            (BASE_PATH / <session> / world_timestamps_0start.npy).

        Returns
        -------
        dict
            session -> SessionClipList
        """
        groups = _group_by_session(self)
        if (native_timestamps is not None) and (not isinstance(native_timestamps, dict)) and (len(groups) > 1):
            raise ValueError('Give `native_timestamps` as a dict of session -> timestamps for several sessions')
        out = {}
        for session in _sort_unique(groups.keys()):
            clips = groups[session]
            if native_timestamps is None:
                timestamps = np.load(BASE_PATH / str(session) / 'world_timestamps_0start.npy')
            elif isinstance(native_timestamps, dict):
                timestamps = native_timestamps[session]
            else:
                timestamps = native_timestamps
            out[session] = SessionClipList([(c.onset, c.offset) for c in clips], timestamps,
                                           session=session, tags=[c.tag for c in clips])
        return out
    
    def save(self, fname=None, path=SAMPLE_PATH):
        """Save clips (onset, offset, session, tag) to `path / fname`
        (by default `<name>.npz` in SAMPLE_PATH)"""
        if fname is None:
            fname = f'{self.name}.npz'
        fpath = pathlib.Path(path) / fname
        to_save = [dict((k, getattr(x, k)) for k in ['onset', 'offset', 'session','tag']) for x in self]
        np.savez(fpath, **utils.dictlist_to_arraydict(to_save))
    
    
    @classmethod
    def load(cls, fpath):
        if not isinstance(fpath, pathlib.Path):
            fpath = pathlib.Path(fpath)
        ob = cls.__new__(cls)
        cc = utils.arraydict_to_dictlist(dict(np.load(fpath, allow_pickle=True)))
        ob.__init__([SessionClip(**c) for c in cc], name=fpath.stem)
        return ob

    def __len__(self):
       return(len(self.clip_list))
    
    def __getitem__(self, i):
        return self.clip_list[i]
    
    def __iter__(self):
        return iter(self.clip_list)

    def __repr__(self):
        return f'ClipList {self.name}: {len(self)} clips from {len(self.sessions)} sessions'
    

def _clean_str(x): 
    return x.lower().strip().replace('_',' ')    

def parse_csv(folder):
    """parse csv file for folder
    
    folder is a full path or pathlib.Path instance pointing to a folder in VEDB official folder,
    or a full path to the csv file to be read. If file name is not included, it is assumed to be
    the same name as the folder (which it should be) + .csv
    
    
    """
    if not isinstance(folder, pathlib.Path):
        folder = pathlib.Path(folder)
    if not folder.exists():
        raise ValueError('`folder` input does not exist')
    if 'csv' not in folder.name:
        fname = folder / f'{folder.name}.csv'
    else:
        fname = copy.copy(folder)
        folder = folder.parent
    time_file = str(folder / 'world_timestamps_0start.npy')
    world_time = np.load(time_file) 
    n = len(world_time)
    assumed_fps = 30 # ?
    with fname.open() as fid:
        lines = fid.readlines()
        locations = []
        tasks = []
        # Initialize frames to start at beginning (first frame)
        # all other frames are LAST frame for an activity
        frames = [0]
        
        for line in lines[1:]:
            parsed_line = line.strip().strip(',').split(',')
            # Check for all-commas, other manual errors
            if len(parsed_line) == 1:
                if parsed_line[0].strip(' ') == '':
                    continue
                else:
                    raise ValueError(f'Unclear line parsing for {fname}')
            elif len(parsed_line) == 4:
                # Does not contain frame
                continue
            elif len(parsed_line) == 5:
                _, minute, loc, task, tmp = parsed_line
                if not (('frame' in tmp.lower()) or (tmp == '')):
                    raise ValueError(f'Unclear line parsing for {fname}; no "frame" in last pos')
                fr = int(tmp.split('.')[0][5:])
            else:
                raise ValueError(f'Unclear line parsing for {fname}')
            frames.append(fr)
            locations.append(_clean_str(loc))
            tasks.append(_clean_str(task))
                
    clips = []
    for st, fin, loc_, lab_ in zip(frames[:-1], frames[1:], locations, tasks):
        st = np.minimum(st, n-1)
        fn = np.minimum(fin, n-1)
        clip = SessionClip.from_indices(st, fn, world_time, session=str(folder.name), tag=f'{loc_}:{lab_}')
        clips.append(clip)
    
    return clips

def _check_paths(fpath):
    file_check = os.listdir(fpath)
    missing_files = []
    for fname in REQUIRED_FILES:
        if isinstance(fname, tuple):
            found = False
            for fnm in fname:
                found = fnm in file_check
                if found:
                    break
            if not found:
                missing_files.append(fname[-1])
        else:
            # TEMP: Use timestamps_0start only
            if 'timestamps' in fname:
                fname = fname.replace('timestamps', 'timestamps_0start')
            if not fname in file_check:
                missing_files.append(fname)
    return missing_files

def save_clips(clips, fname):
    to_save = [dict((k, getattr(x, k)) for k in ['onset', 'offset', 'session','tag']) for x in clips]
    np.savez(fname, **utils.dictlist_to_arraydict(to_save))

def load_clips(fname):
    cc = utils.arraydict_to_dictlist(dict(np.load(fname, allow_pickle=True)))
    return [SessionClip(**c) for c in cc]



# Maybe move me: 

def gaze_rect(gaze_position, hdim, vdim, ax=None, linewidth=1, edgecolor='r', **kwargs):
    if ax is None:
        ax = plt.gca()
    # Create a Rectangle patch
    gp = gaze_position - np.array([hdim, vdim]) / 2
    rect = patches.Rectangle(gp, hdim, vdim,
                             facecolor='none',
                             edgecolor=edgecolor,
                             linewidth=linewidth,
                             **kwargs)
    # Add the patch to the Axes
    rh = ax.add_patch(rect)
    return rh


