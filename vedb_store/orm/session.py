from .mappedclass import MappedClass
from .. import utils
from .. import options
import file_io
from collections import OrderedDict
import datetime
import warnings
import pathlib
import hashlib
import numpy as np
import copy
import os

from ..utils import get_frame_indices, get_time_split, SESSION_FIELDS, load_pipeline_elements, onoff_from_binary

BASE_PATH = pathlib.Path(options.config.get('paths', 'vedb_directory')).expanduser()
PROC_PATH = pathlib.Path(options.config.get('paths', 'proc_directory')).expanduser()
SESSION_INFO = dict(np.load(BASE_PATH / 'session_info_wip.npz'))


import file_io
#from vedb_gaze.visualization import show_ellipse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation, patches, colors, gridspec

def show_ellipse(ellipse, img=None, ax=None, center_color='r', **kwargs):
    """Show opencv ellipse in matplotlib, optionally with image underlay

    Parameters
    ----------
    ellipse : dict
        dict of ellipse parameters derived from opencv, with fields:
        * center: tuple (x, y)
        * axes: tuple (x length, y length)
        * angle: scalar, in degrees
    img : array
        underlay image to display
    ax : matplotlib axis
        axis into which to plot ellipse
    kwargs : passed to matplotlib.patches.Ellipse
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

# Pipeline defaults
# pipeline_default = dict(
#     calibration_marker = 'find_concentric_circles-circles_halfres',
#     calibration_split = 'split_circles',
#     calibration_cluster = 'cluster_circles',
#     validation_marker = 'find_checkerboards-checkerboard_halfres_%ssquares',
#     validation_split = 'split_checkerboards',
#     validation_cluster = 'cluster_checkerboards',
#     pupil = 'pylids_eyelids_pupils_v2',
#     pupil_detrending = 'estimate_slippage-full_eyelid_shape',
#     calibration = 'monocular_tps_cv_cluster_median_conf75_cut3std', # monocular_tps_default'
#     gaze_mapping = 'default_mapper', 
#     error = '',
# )
pipeline_default = dict(
                  pupil_tag='pylids_pupils_eyelids_v2',
                  pupil_detrend_tag=None,
                  calibration_marker_tag='circles_halfres',
                  calibration_split_tag=None,
                  calibration_cluster_tag='cluster_circles',
                  validation_marker_tag='checkerboard_halfres_4x7squares',
                  validation_split_tag=None, 
                  validation_cluster_tag='cluster_checkerboards',
                  calibration_tag='monocular_tps_cv_cluster_median_conf75_cut3std',
                  gaze_tag='default_mapper',
                  error_tag='smooth_tps_cv_clust_med_outlier4std_conf75', 
                  calibration_epoch=0,
)


def make_file_strings(
        pupil_tag='pylids_pupils_eyelids_v2',
        pupil_detrend_tag=None,
        calibration_marker_tag='circles_halfres',
        calibration_split_tag=None,
        calibration_cluster_tag='cluster_circles',
        validation_marker_tag='checkerboard_halfres_4x7squares',
        validation_split_tag=None, 
        validation_cluster_tag='cluster_checkerboards',
        calibration_tag='monocular_tps_cv_cluster_median_conf75_cut3std',
        gaze_tag='default_mapper',
        error_tag='smooth_tps_cv_clust_med_outlier4std_conf75', 
        calibration_epoch=0,
        # Extra
        eye=None,
        fov_str=None,
        validation_epoch=0):
        # Hashes of inputs for steps with too many inputs for a_b_c type filename construction
    if fov_str is None:
        fov_str = '%s'
    if eye is None:
        eye = '%s'
    calibration_args = [x for x in [calibration_marker_tag, calibration_split_tag, \
                                       calibration_cluster_tag, f'epoch{calibration_epoch:02d}', \
                                       pupil_tag, pupil_detrend_tag] if x is not None]
    # '-' will mess up later parsing of file names, so replace; this *might* make hashes non-unique, but is most likely to be fine.
    calibration_input_hash = hashlib.blake2b(('-'.join(calibration_args)).replace('-','0').encode(), digest_size=10).hexdigest()
    error_args = [x for x in [calibration_marker_tag, calibration_split_tag, \
                                       calibration_cluster_tag, f'epoch{calibration_epoch:02d}', \
                                       pupil_tag, pupil_detrend_tag, \
                                       calibration_tag, gaze_tag,
                                       validation_marker_tag, validation_split_tag, validation_cluster_tag, \
                                       ] if x is not None]
    error_input_hash = hashlib.blake2b(('-'.join(error_args)).replace('-','0').encode(), digest_size=10).hexdigest()
    out = dict(
        pupil_file = f'pupil-{eye}-{pupil_tag}.npz',
        gaze_file = f'gaze-{eye}-{gaze_tag}-{calibration_tag}-{calibration_input_hash}.npz',
        error_file = f'error-%s-{error_tag}_{fov_str}-{error_input_hash}-epoch{validation_epoch:02d}.npz',
        )
    return out


# input_hash = hash('-'.join([pipeline[x] for x in ['calibration_marker','calibration_filter',
#                                                   'validation_marker','validation_filter',
#                                                   'pupil','detrending',
#                                                   ]]))


 
# # Slippage correction
# pupil_detrend_path = f'pupil_detrended-%s-{pipeline['detrending']}.npz'
# calibration_marker_path = f'markers-calibration-{calibration_marker_string}-epochall.npz' # epoch all - revisit?
# validation_marker_path = f'markers-validation-{validation_marker_string}-epoch%02d.npz'
# calibration_path = f'calibration_%s_{hash()}.npz'
# error_str = 'error_%s.npz'
# gaze_str = 'gaze_%s.npz'

class Session(MappedClass):
    """Representation of a VEDB recording session.
    
    Contains paths to all relevant files (world video, eye videos, etc.)
    and means to load them, as well as meta-data about the session.
    """
    def __init__(self, 
            folder=None, 
            subject=None, 
            date=None,
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
        self.date = date
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
        self._temp_fields = [
            'path',
            'paths',
            'clips',
            'datetime', 
            'world_time', 
            'recording_duration']
        # Fields that are other database objects
        self._db_fields = []
        # This might be time consuming for large queries...
        self.load_clips()

    def load_clips(self):
        clip_file = pathlib.Path(self.path) / f'{self.folder}.csv'
        if clip_file.exists():
            self.clips = parse_csv(clip_file)
        
    def load_gaze_pipeline(self, pipeline='latest', is_verbose=0):
        """load all elements of a gaze pipeline

        Parameters
        ----------
        pipeline : str, optional
            tag (name) of pipeline, by default 'latest', which (medium intelligently)
            finds latest greatest estimate of gaze
        is_verbose : int or bool
            False or 0 : print nothing
            True or 1 : print whether each element was found
            2+ : print all above and database query results
        """
        if self.dbi is None:
            warnings.warn('self.dbi must be active database interface for this to work')
            return None
        # Get list of pipeline keys
        pl = self.dbi.query(1, type='ParamDictionary', fn='vedb_gaze.pipelines.make_pipeline', tag=pipeline)
        ple = load_pipeline_elements(self, dbi=self.dbi, is_verbose=is_verbose, **pl.params)
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
        if (len(missing_files) > 0) & raise_error:
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
            
        # Set date    (& be explicit about what constitutes date)    
        try:
            # Assume year_month_day_hour_min_second for date specification in folder title
            session_date = datetime.datetime.strptime(folder.name, '%Y_%m_%d_%H_%M_%S')
            date = folder.name
        except:
            # Switch on verbosity?
            date = None
            print('Folder name not parseable as a date')

        ob.__init__(dbi=dbinterface,
              subject=subject,
              folder=folder.name,
              date=date
              )
        
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
        # lazy loading
        self.gaze_paths = None
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
    
    def load_gaze_pipeline(self, pipeline=pipeline_default, clock='native', eye=('left','right')):
        """Load specific gaze pipeline. Allows for non-default gaze to be loaded.
        
        """
        if self.session is None:
            return None
        self.gaze_paths = make_file_strings(pipeline)
        self._pupil = dict((lr, np.load(PROC_PATH / self.session /self.gaze_paths['pupil']%lr)) for lr in eye)
        if clock is not 'native':
            for e in eye:
                self._pupil[e] = match_time_points([dict(timestamp=self.world_time), self._pupil[e]])
        self._gaze = dict((lr, np.load(PROC_PATH / self.session /self.gaze_paths['gaze']%lr)) for lr in eye)
        #self._error = dict((lr, np.load(self.gaze_paths['gaze']%lr)) for lr in eye)

    @property
    def pupil(self):
        if self.gaze_paths is None:
            self.load_gaze_pipeline()
        if self._pupil is None:
            pupil_file_left = PROC_PATH / self.session / (self.gaze_paths['pupil']%'left')
            pupil_file_right = PROC_PATH / self.session / (self.gaze_paths['pupil']%'right')
            self._pupil = {}
            if pupil_file_left.exists():
                self._pupil['left'] = self(dict(np.load(pupil_file_left, allow_pickle=True)))
            if pupil_file_right.exists():
                self._pupil['right'] = self(dict(np.load(pupil_file_right, allow_pickle=True)))
        return self._pupil
    
    @property
    def gaze(self):
        if self._gaze is None:
            gaze_file_left = PROC_PATH / self.session / gaze_path%'left'
            gaze_file_right = PROC_PATH / self.session / gaze_path%'right'
            gaze = {}
            if gaze_file_left.exists():
                gaze['left'] = self(dict(np.load(gaze_file_left)))
            if gaze_file_right.exists():
                gaze['right'] = self(dict(np.load(gaze_file_right)))
            self._gaze = gaze
        return self._gaze

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
            fname = BASE_PATH / self.session / 'worldPrivate.mp4'
            ftime = BASE_PATH / self.session / 'world_timestamps_0start.npy'
            world_time = np.load(ftime)
            data = file_io.load_video(fname, frames=self.indices(world_time), **kwargs)
            time = world_time[self.binary(world_time)]
            return time, data
        elif data_type == 'gaze':
            out = load_gaze(self.session, **kwargs)
            return self(out)
        elif data_type == 'odometry':
            odo = file_io.load_msgpack((BASE_PATH / self.session / 'odometry.pldata'))
            return self(odo)
        elif data_type in ('eye_left', 'eye_right'):
            lr = '1' if 'left' in data_type else '0'
            fname = BASE_PATH / self.session / f'eye{lr}_blur.mp4'
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

    def make_gaze_animation(self,
                            rect_size=(600,600),
                            pupil_str = 'pupil_detection-%s-pylids_eyelids_pupils_v2.npz',
                            gaze_str = 'gaze-%s-default_mapper-monocular_tps_cv_cluster_median_conf75_cut3std-c0eb1e655a0f58e7905b.npz',
                            fps=30, 
                            world_size_factor=0.25,
                            hspace=0.1,
                            wspace=None,
                            eye_left_color=(1.0, 0.5, 0.0),  # orange
                            eye_right_color=(0.0, 0.5, 1.0),  # cyan
                            session_info=SESSION_INFO,
                            raise_error=False,
                        ):
        """Make radical gaze animation"""
        global eye_left_frame
        global eye_left_image
        global eye_right_frame    
        global eye_right_image
        eye_video_size = 400 # x 400, square
        tmp = utils.arraydict_to_dictlist(session_info)
        si_dict = dict((x['folder'], x) for x in tmp)
        si = si_dict[self.session]
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
            
            _, vh, vw, _ = file_io.list_array_shapes(ses.paths['world_camera'][1])
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
                #print('rendering pupil left')
                pupil_left = dict(np.load(pl, allow_pickle=True))
                ellipse_data_left = dict((k, np.array(v) / eye_video_size)
                                        for k, v in pupil_left['ellipse'][0].items())
                pupil_left_eh, pupil_left_dh = show_ellipse(ellipse_data_left,
                                        center_color=eye_left_color,
                                        facecolor=eye_left_color +
                                        (0.5,),
                                        ax=ax_eye_left)
                #print(pupil_left_eh)

            if include_right_eye:
                success, eye_right_image = eye_right_vid.VideoObj.read()
                eye_right_h = ax_eye_right.imshow(eye_right_image, extent=[0, 1, 1, 0])
                
            if pr.exists():
                #print('rendering pupil right')
                pupil_right = dict(np.load(pr, allow_pickle=True))
                ellipse_data_right = dict((k, np.array(v) / eye_video_size)
                                        for k, v in pupil_right['ellipse'][0].items())
                pupil_right_eh, pupil_right_dh = show_ellipse(ellipse_data_right,
                                        center_color=eye_right_color,
                                        facecolor=eye_right_color +
                                        (0.5,),
                                        ax=ax_eye_right)
                #print(pupil_right_eh)
                
            ax_eye_left.axis([1, 0, 1, 0]) # [0,1, 0, 1]
            ax_eye_left.set_xticks([])
            ax_eye_left.set_yticks([])
            
            ax_eye_right.axis([0, 1, 0, 1]) #[1, 0, 1, 0]
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
                    #print(eye_left_h)
                    eye_left_h.set_array(np.zeros((400,400), dtype=np.uint8)) #np.zeros_like(eye_left_ds[0]))
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
                        #print("GAAAAAA")
                    to_return.extend([gaze_rect_lh, gaze_dot_lh, gc_h])
                    
                if gr.exists():
                    gaze_rect_rh.set_xy(gaze_right['norm_pos'][eye_right_frame] - np.array([rect_width/2, rect_height/2]))
                    gaze_dot_rh.set_offsets(gaze_right['norm_pos'][eye_right_frame])
                    to_return.extend([gaze_rect_rh, gaze_dot_rh])

                
                return to_return



            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1/fps * 1000, blit=True)
        except:
            eye_left_vid.VideoObj.release()
            eye_right_vid.VideoObj.release()
            raise
        return anim

        
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
            out = np.any(np.vstack([self.binary(timestamps=timestamps,
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
        out = onoff_from_binary(ii, return_duration=False)
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
        onoff_indices = onoff_from_binary(binary, return_duration=False)
        onoffs = [native_timestamps[ii] for ii in onoff_indices]
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob
    
    @classmethod
    def from_clips(cls, list_of_clips, native_timestamps):
        ob = cls.__new__(cls)
        # Require that all clips are from same session. For now, clip lists must tbe same session.
        session = list_of_clips[0].session
        assert all([x.session == session for x in list_of_clips]),\
              'SessionClip objects in ClipList must be from same session!'
        onoffs = [(x.onset, x.offset) for x in list_of_clips]
        tags = [x.tag for x in list_of_clips]
        ob.__init__(onoffs, native_timestamps, session=session, tags=tags)
        return ob
    
    def __sub__(self, cliplist):
        new_clips = []
        #chk = cliplist.binary(self.native_timestamps)
        for clip in self:
            onset_btw = any([(other_clip.onset >= clip.onset) &\
                              (other_clip.onset <= clip.offset) for other_clip in cliplist])
            offset_btw = any([(other_clip.offset >= clip.onset) &\
                               (other_clip.offset <= clip.offset) for other_clip in cliplist])
            if not onset_btw | offset_btw:
                new_clips.append(clip)
        if len(new_clips) == 0:
            return []
        onoffs = [(x.onset, x.offset) for x in new_clips]
        tags = [x.tag for x in new_clips]
        return ClipList(onoffs, self.native_timestamps, session=self.session, tags=tags)
        #return ClipList.from_binary(bb, self.native_timestamps, self.session)

    def __add__(self, cliplist):
        a = self.binary(self.native_timestamps)
        b = cliplist.binary(self.native_timestamps)
        return ClipList.from_binary(a | b, self.native_timestamps, session=self.session) #, tags=self.tags)

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
