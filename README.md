# vedb-store

Session-level data access for the Visual Experience Database (VEDB).

vedb-store gives you objects for VEDB recording sessions and for time
intervals within them, and loads raw data streams (world and eye videos,
odometry) and the outputs of the [vedb-gaze](https://github.com/vedb/vedb-gaze)
gaze pipeline (pupils, gaze, gaze error) for them.

| Class | What it represents |
| --- | --- |
| `Session` | one recording session (a folder in the VEDB directory) |
| `SessionClip` | a time interval within one session, e.g. a labeled task |
| `SessionClipList` | several clips from the *same* session, on one timeline; supports `+`, `-`, `&`, `invert`, `dilate` |
| `ClipList` | clips from *many* sessions, e.g. a sample of VEDB to analyze; supports `+`, `-`, `&`, `dilate`, `save` / `load` |

All times are in seconds, zero-referenced to the start of the session (the
`*_timestamps_0start.npy` files).

## Installation

vedb-store builds on vedb-gaze, so install vedb-gaze first (see its README),
then vedb-store:

```bash
pip install git+https://github.com/vedb/vedb-gaze.git   # or: pip install -e /path/to/vedb-gaze
pip install -e .                                          # from this repository
```

`environment.yml` creates a conda environment with both packages and their
dependencies (`conda env create -f environment.yml`).

## Configuration

Data locations are set in the `[paths]` section of the user config at
`~/.config/vedb_store/options.cfg` (Linux; created on first import from
`vedb_store/defaults.cfg`):

| Key | Meaning | If empty |
| --- | --- | --- |
| `vedb_directory` | folder with VEDB session folders, `session_info.npz` and `session_error.npz` | `base_dir` from the vedb-gaze config |
| `proc_directory` | folder with processed outputs (`<proc>/<session>/*.npz`) | `proc_dir` from the vedb-gaze config |
| `sample_directory` | default folder for `ClipList.save` | current directory |

The defaults are empty, so vedb-store uses the same folders as vedb-gaze
(`~/.config/vedb-gaze/options.cfg`). Set a key only to override that. Older
`options.cfg` files are full copies of the old defaults and still contain
explicit paths; delete those lines to follow the vedb-gaze config instead.

`session_info.npz` (per-session field of view and validation checkerboard
size) is read on first use, not on import, so `import vedb_store` works
without access to the data.

## Usage

```python
from vedb_store import Session, SessionClip, ClipList

ses = Session('2021_02_27_10_12_44')          # folder name in vedb_directory
tt, frames = ses.load('world_camera', time_idx=(60, 65), size=0.25)
tt, odo = ses.load('odometry', time_idx=(60, 65))        # processed (see below)
tt, pitch = ses.load('odometry:pitch', time_idx=(60, 65))

# Gaze-pipeline outputs, for the default vedb-gaze tags
gaze = ses.gaze                                # {'left': ..., 'right': ...}
err = ses.error                                # validation epoch 0
best = ses.load_gaze(eye='best', clock='world_time')

# Another pipeline: give only the tags that differ from the defaults,
# with vedb-gaze's names ('pupil', 'calibration', 'error', ...)
ses.load_gaze_pipeline(pipeline=dict(calibration='monocular_tps_default'))

# Clips (from <session>/<session>.csv, if present)
for clip in ses.clips:
    print(clip.tag, clip.duration)
clip = ses.clips[0]
gaze_clip = clip.gaze_best                     # gaze of the eye with lower error
odo_clip = clip.load('odometry')

# Samples across sessions
samples = ClipList([c for c in ses.clips if c.task == 'walking'], name='walking')
samples.save()                                 # sample_directory / walking.npz
samples = ClipList.load('walking.npz')
```

## Relation to vedb-gaze

vedb-store does not keep its own copy of anything vedb-gaze defines:

- **File names** of pipeline outputs come from
  `vedb_gaze.utils.make_file_strings`, and the default processing tags from
  vedb-gaze's config (`vedb_store.session.pipeline_default` is
  `vedb_gaze.utils.defaults`). Loading always matches what
  `vedb_gaze.pipelines.pipeline_vedb` writes. Old-style pipeline keys ending
  in `_tag` (e.g. `pupil_tag`) still work, with a DeprecationWarning.
- **Data folders** default to vedb-gaze's `base_dir` / `proc_dir` (above).
- **Utilities** in `vedb_store.utils` such as `onoff_from_binary`,
  `match_time_points` and `get_frame_indices` are vedb-gaze's functions.
- **Odometry** is loaded with `vedb_gaze.odometry.load_odometry`
  (resampled, smoothed, with `pitch` / `roll` / `yaw` in degrees). Pass
  `processed=False` to `Session.load` / `SessionClip.load` for the raw
  `odometry.pldata` records.
- **Best eye** (`Session.load_gaze(eye='best')`, `SessionClip.gaze_best`)
  is the eye with lower `gaze_err_weighted` in the pipeline's error files.

vedb-gaze does not import vedb-store. Its plotting functions that read video
frames (e.g. `vedb_gaze.visualization.plot_eye_at_marker`,
`plot_session_qc(..., session=ses)`) accept a `vedb_store.Session`.

`vedb_store.orm` is a deprecated alias for `vedb_store.session`, kept so old
imports keep working.

## License

GPL-3.0 (see `LICENSE`).
