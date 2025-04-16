from .mappedclass import MappedClass
from .. import options
import file_io
from collections import OrderedDict
import datetime
import warnings
import pathlib
import numpy as np
import copy
import os

from ..utils import get_frame_indices, get_time_split, SESSION_FIELDS, load_pipeline_elements

BASE_PATH = pathlib.Path(options.config.get('paths', 'vedb_directory')).expanduser()


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


class Session(MappedClass):
    """Representation of a VEDB recording session.
    
    Contains paths to all relevant files (world video, eye videos, etc.)
    and means to load them, as well as meta-data about the session.
    """
    def __init__(self, 
            folder=None, 
            subject=None, 
            task=None, 
            location=None, 
            fov=None,
            # Database bureaucracy
            type='Session',
            dbi=None, 
            _id=None, 
            _rev=None,
            **kwargs):
        """Class for a data collection session for vedb project

        """
        self.type = 'Session'
        self.folder = folder
        self.subject=subject
        self.fov=fov
        self.dbi = dbi
        self._id = _id
        self._rev = _rev
        self._base_path = BASE_PATH
        self._path = None
        self._paths = None
        self._features = None
        self._world_time = None
        self._recording_duration = self.world_time[-1] - self.world_time[0]
        # Introspection
        # Will be written to self.fpath (if defined)
        self._data_fields = []
        # Constructed on the fly and not saved to docdict
        self._temp_fields = ['path',
                       'paths',
                       'datetime', 
                       'world_time', 
                       'recording_duration']
        # Fields that are other database objects
        self._db_fields = []
    
    def load_gaze_pipeline(self, pipeline='latest'):
        """load all elements of a gaze pipeline

        Parameters
        ----------
        pipeline : str, optional
            tag (name) of pipeline, by default 'latest', which (medium intelligently)
            finds latest greatest estimate of gaze
        """
        if self.dbi is None:
            warnings.warn('self.dbi must be active database interface for this to work')
            return None
        # Get list of pipeline keys
        pl = self.dbi.query(1, type='ParamDictionary', fn='vedb_gaze.pipelines.make_pipeline', tag=pipeline)
        ple = load_pipeline_elements(self, dbi=self.dbi, **pl.params)
        return ple

    def load_gaze(self, pipeline='latest', clock='native', time_idx=None):
        """Load estimate of gaze based on particular pipeline tag

        Parameters
        ----------
        pipeline : str, optional
            name of pipeline, by default 'latest', which (medium intelligently)
            pulls latest estimate of gaze
        clock : str, optional
            which timestamps gaze should have. Default is 'native', which means at ~120 hz 
            (native eye camera temporal resolution). 'world' specifies that gaze should be
            matched to nearest world timestamp (or averaged over a window according to
            `kwargs` that are passed on to `match_timepoints()`)
        """
        ple = self.load_gaze_pipeline(pipeline=pipeline)
        return ple['gaze']

    def load(self, data_type, time_idx=None, frame_idx=None, **kwargs):
        """
        Parameters
        ----------
        data_type : string
            one of: 'world_camera', 'tracking_camera', 'eye_camera', 
            some data types have multiple sub-components; these can be accessed
            with colon syntax, e.g. data_type='odometry:location'
        time_idx : tuple
            (start time, end time) in seconds since the start of the session.
            Note that some streams (e.g. world_camera) may not start right at
            0 seconds, because they take some time to get started
        frame_idx : tuple
            (start_frame, end_frame)
        kwargs : dict
            passed to relevant loading function. 
            Note that for movie data, you can reshape and change the color format
            by using e.g. `size=(90, 120)` or `color='gray'` (see file_io.load_mp4)
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
        return file_io.VideoCapture(self.paths[stream][1])
    
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
            base_path = self._resolve_sync_dir(self.path)
            for fnm, nm in zip(to_find, names):
                if isinstance(fnm, tuple):
                    ff = 'no_file.nope'
                    for f in fnm:
                        #print('looking for',str(base_path / f))
                        if (base_path / f).exists():
                            ff = copy.copy(f)
                            #print('FOUND IT!')
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
                    #print(f'Adding {nm} to paths')
                    _paths[nm] = (timestamp_path, data_path)
                else:
                    print(f'failed for {nm}:', timestamp_path, data_path)
            self._paths = _paths
        return self._paths
    
    # @property
    # def features(self):
    #     if self._features is None:
    #         # perhaps sort these by tag?
    #         self._features = self.dbi.query(type='Features', session=self._id)
    #     return self._features

    @classmethod
    def from_folder(cls, folder, dbinterface=None, subject=None, raise_error=True, db_save=False, load_label_csv=True):
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
        #print('\n>>> Importing folder %s'%folder)
        if ~isinstance(folder, pathlib.Path):
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
        if len(missing_files) > 0:
            raise ValueError(f'Missing files: {missing_files}\n')
        # Check for presence of folder in database if we are aiming to save session in database
        if db_save:
            check = dbinterface.query(type='Session', folder=folder.name)
            if len(check) > 0:
                print('SESSION FOUND IN DATABASE.')
                return check[0]
            elif len(check) > 1:
                raise Exception('More than one database session found with this date!')                
        # VEDB specific things: folder name as date, csv file for task, location labels
        # look for session csv for vedb
        if load_label_csv:
            clips = parse_csv(folder / f'{folder.name}.csv')
        # Set date    (& be explicit about what constitutes date)    
        try:
            # Assume year_month_day_hour_min_second for date specification in folder title
            session_date = datetime.datetime.strptime(folder.name, '%Y_%m_%d_%H_%M_%S')
        except:
            # Switch on verbosity?
            session_date = None
            print('Folder name not parseable as a date')

        ob.__init__(dbi=dbinterface,
              subject=subject,
              folder=folder.name,
              date=session_date
              )
        if load_label_csv:
            ob.clips = clips
        return ob


class SessionClip(object):
    """Clip of a session."""
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
        #self.native_timestamps = native_timestamps
    
    @classmethod
    def from_indices(cls, onset_i, offset_i, timestamps, session=None, tag=None):
        ob = cls.__new__(cls)
        ob.__init__(timestamps[onset_i], timestamps[offset_i], session=session, tag=tag)
        return ob
    
    @property
    def duration(self,):
        return self.offset - self.onset
    
    @property
    def task(self,):
        if self.tag is None:
            return None
        if ':' in self.tag:
            return self.tag.split(':')[1]
    
    @property
    def location(self,):
        if self.tag is None:
            return None
        if ':' in self.tag:
            return self.tag.split(':')[0]    
        
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
    
    def __call__(self, inpt, key=None, pre=0, post=0, resample_function=None, **kwargs):
        assert isinstance(inpt, dict), "Input must be dictionary!"
        assert 'timestamp' in inpt, "dict input must have 'timestamp' field!"
        tt = inpt['timestamp']
        ii = self.binary(tt, pre=pre, post=post)
        output = {}
        if key is None:
            keys = list(inpt.keys())
        else:
            keys = [key]
        for k in keys:
            if isinstance(inpt[k], dict):
                # Skip dictionaries for now
                pass
            # RESAMPLE if asked
            output[k] = inpt[k][ii]
        if key is not None:
            return output[k]
        else:
            return output
        
    def load(self, data_type, **kwargs):
        if self.session is None:
            raise ValueError('Cannot load data with clip `session` set')
        if data_type == 'world_camera':
            fname = fbase_official / self.session / 'worldPrivate.mp4'
            ftime = fbase_official / self.session / 'world_timestamps_0start.npy'
            world_time = np.load(ftime)
            data = file_io.load_video(fname, frames=self.indices(world_time), **kwargs)
            time = world_time[self.binary(world_time)]
            return time, data
        elif data_type == 'gaze':
            out = load_gaze(self.session, **kwargs)
            return self(out)
        elif data_type == 'odometry':
            odo = dict(np.load(fbase_data / self.session / 'odometry.npz'))
            return self(odo)
        elif data_type in ('eye_left', 'eye_right'):
            lr = '1' if 'left' in data_type else '0'
            fname = fbase_official / self.session / f'eye{lr}_blur.mp4'
            ftime = fbase_official / self.session / f'eye{lr}_timestamps_0start.npy'
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
        """Choose `n_samples` non-overlapping random samples within this 

        """
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
    
        
    def __repr__(self,):
        return (f'Clip for {self.session}\n'
                f'> {self.duration:.2f}s, from {self.onset:.2f}-{self.offset:.2f}\n'
                f'> {self.tag}'
               )
    

class ClipList(object):
    def __init__(self, onsets_offsets, native_timestamps, session=None, tags=None):
        """List of clips
        
        all times passed to this and its methods must be normalized to session start
        i.e. starting when the first data stream for a session came online.
        
        """
        self.clip_list = []
        self.native_timestamps = native_timestamps
        self.session = session
        if not isinstance(onsets_offsets, np.ndarray):
            onsets_offsets = np.asarray(onsets_offsets)
        self.native_indices = onsets_offsets[:,:2]
        for j, onoff in enumerate(onsets_offsets):
            if len(onoff) == 2:
                onset, offset = onoff
            elif len(onoff) == 3:
                onset, offset, duration = onoff
            if tags is None:
                tag = None
            else:
                tag = tags[j]
            self.clip_list.append(SessionClip(onset, offset, session=session, tag=tag))
    
    def binary(self, timestamps=None, comparison_type=('>=', '<'), pre=0, post=0, any_way=False):
        """Return binary version of this list over full duration of 
        `native_timestamps`
        
        i.e., convert this ClipList to a vector of True values for native 
        timestamps *during* clips and False values for native timestamps 
        *outside* of clips.
        """
        if timestamps is None:
            timestamps = self.native_timestamps
        if any_way:
            out = np.any(np.vstack([clip.binary(timestamps=timestamps,
                                                comparison_type=comparison_type,
                                                pre=pre,
                                                post=post)
                                    for clip in self]), axis=0)
        else:
            out = np.zeros_like(timestamps)
            for clip in self:
                st, fin = clip.indices(timestamps, comparison_type=comparison_type)
                if st is not None:
                    out[st:fin] = 1
            out = out > 0
        return out

    def indices(self, timestamps=None, comparison_type=('>=', '<')):
        if timestamps is None:
            timestamps = self.native_timestamps
        ii = self.binary(timestamps=timestamps, comparison_type=comparison_type)
        out = vedb_gaze.utils.onoff_from_binary(ii, return_duration=False)
        return np.asarray(out)
    
    def times(self, include_duration=False):
        return np.asarray([clip.times(include_duration) for clip in self.clip_list])
        
    def filter_duration(self, min_duration=0, max_duration=np.inf):
        """Durations in seconds"""
        onoff_times = self.times(include_duration=True)
        onoffs_filtered = [x for x in onoff_times if (x[2] > min_duration) and (x[2]  < max_duration)]
        return ClipList(onoffs_filtered, self.native_timestamps, session=self.session, tags=self.tags)

    def dilate(self, pre=0, post=0, merge=False):
        """dilate time for each clip. Times in seconds.

        """
        if merge:
            ii = self.binary(pre=pre, post=post)
            out = ClipList.from_binary(ii, self.native_timestamps, session=self.session)
        else:
            # Don't overwrite self
            mn = self.native_timestamps.min()
            mx = self.native_timestamps.max()
            onoff_clips = [(x.dilate(pre=pre, post=post, tlimits=(mn, mx), )) \
                            for x in self]
            onoff_times = [(x.onset, x.offset) for x in onoff_clips]
            out = ClipList(onoff_times, self.native_timestamps, session=self.session, tags=self.tags)
        return out

    def invert(self, **kwargs):
        out = ClipList.from_binary(~self.binary(**kwargs), self.native_timestamps, session=self.session)
        # Kill any 1-frame clips
        out.clip_list = [clip for clip in out if clip.duration > 0]
        return out

    @property
    def tags(self):
        return [x.tag for x in self]

    @classmethod
    def from_indices(cls, onoff_indices, native_timestamps, session=None, tags=None):
        ob = cls.__new__(cls)
        onoffs = [native_timestamps[ii] for ii in onoff_indices]
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob

    @classmethod
    def from_binary(cls, binary, native_timestamps, session=None, tags=None):
        ob = cls.__new__(cls)
        onoff_indices = vedb_gaze.utils.onoff_from_binary(binary, return_duration=False)
        onoffs = [native_timestamps[ii] for ii in onoff_indices]
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob
    
    def __sub__(self, cliplist):
        new_clips = []
        #chk = cliplist.binary(self.native_timestamps)
        for clip in self:
            #b = clip.binary(self.native_timestamps)
            #if not any(b & chk):
            #    new_clips.append(b)
        #bb = np.any(np.vstack(new_clips), axis=0)
        #return ClipList.from_binary(bb, self.native_timestamps, self.session)

            onset_btw = any([(other_clip.onset >= clip.onset) &\
                              (other_clip.onset <= clip.offset) for other_clip in cliplist])
            offset_btw = any([(other_clip.offset >= clip.onset) &\
                               (other_clip.offset <= clip.offset) for other_clip in cliplist])
            if not onset_btw | offset_btw:
                new_clips.append(clip)
        onoffs = [(x.onset, x.offset) for x in new_clips]
        tags = [x.tag for x in new_clips]
        return ClipList(onoffs, self.native_timestamps, session=self.session, tags=tags)
        #return ClipList.from_binary(bb, self.native_timestamps, self.session)

    def __add__(self, cliplist):
        a = self.binary(self.native_timestamps)
        b = cliplist.binary(self.native_timestamps)
        return ClipList.from_binary(a | b, self.native_timestamps, session=self.session, tags=self.tags)

    def __len__(self):
        return(len(self.clip_list))
    
    def __getitem__(self, i):
        return self.clip_list[i]
    
    def __iter__(self):
        return iter(self.clip_list)
    # Do we want this...?
    # @classmethod
    # def from_clip_list(cls, clip_list, native_timestamps):
    #     ob = cls.__new__(cls)
    #     ob.__init([c.indices(native_timestamps) for c in clip_list], native_timestamps, clip_list[0].session)
    #     return ob


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
            locations.append(loc)
            tasks.append(task)                
                
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
