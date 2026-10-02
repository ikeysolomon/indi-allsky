import copy
import math
from .projection import RADIAL_MIN, RADIAL_MAX

from .calibration import validateCalibration


# (key, cast, min, max) -- solver input limits, not manual save limits.
SOLVER_REQUEST_FIELDS = (
    ('AZIMUTH_ANGLE', float, 0.0, 360.0),
    ('LATITUDE_OFFSET', float, -30.0, 30.0),
    ('LONGITUDE_OFFSET', float, -30.0, 30.0),
    ('IMAGE_CIRCLE_DIAMETER', int, 100, 20000),
    ('OFFSET_X', int, -10000, 10000),
    ('OFFSET_Y', int, -10000, 10000),
)

# every top-level config key that changes the final, post-transform pixel
# space the solve was fit against; if any of these move after a solve,
# LENS_SOLVED must be invalidated or the Milky Way band renders confidently
# in the wrong place
LENS_GEOMETRY_KEYS = (
    'LENS_AZIMUTH',
    'LENS_ALTITUDE',
    'LENS_OFFSET_X',
    'LENS_OFFSET_Y',
    'LENS_IMAGE_CIRCLE',
    'IMAGE_ROTATE',
    'IMAGE_ROTATE_ANGLE',
    'IMAGE_ROTATE_KEEP_SIZE',
    'IMAGE_FLIP_V',
    'IMAGE_FLIP_H',
    'IMAGE_CROP_IMAGE_CIRCLE',
    'IMAGE_CROP_ROI',
    'IMAGE_SCALE',
    'IMAGE_BORDER',
)

# the VIRTUALSKY sub-keys the solver itself writes
LENS_GEOMETRY_VIRTUALSKY_KEYS = (
    'IMAGE_CIRCLE_DIAMETER',
    'LATITUDE_OFFSET',
    'LONGITUDE_OFFSET',
    'OFFSET_X',
    'OFFSET_Y',
    'POINTING_AZIMUTH',
    'PRECESSION',
    'RADIAL_DISTORTION',
    'CALIBRATION_ENABLED',
    'CALIBRATION',
)


def captureLensGeometrySnapshot(config):
    """Snapshot every config value that affects the solved pixel geometry,
    to be compared later via ``invalidateLensSolveIfGeometryChanged``.
    """
    virtualsky = config.get('VIRTUALSKY', {})
    return copy.deepcopy(
        tuple(config.get(key) for key in LENS_GEOMETRY_KEYS)
        + tuple(virtualsky.get(key) for key in LENS_GEOMETRY_VIRTUALSKY_KEYS)
    )


def invalidateLensSolveIfGeometryChanged(config, snapshot):
    """Clear LENS_SOLVED if any geometry key has changed since ``snapshot``
    was captured -- a stale solve is worse than no solve, since the Milky
    Way band would render confidently in the wrong place. Returns True if
    invalidated.
    """
    if not config.get('LENS_SOLVED', False):
        return False

    if captureLensGeometrySnapshot(config) == snapshot:
        return False

    config['LENS_SOLVED'] = False
    return True


def parseSolverRequestValues(data, for_save=False):
    """Validate geometry and optional lens calibration from request JSON.
    Returns (values, None) or (None, error); unknown keys are discarded.
    """
    values = {}
    for key, cast, vmin, vmax in SOLVER_REQUEST_FIELDS + (
            ('POINTING_AZIMUTH', float, 0.0, 360.0), ('LENS_ALTITUDE', float, 0.0, 90.0),
            ('RADIAL_DISTORTION', float, RADIAL_MIN, RADIAL_MAX)):
        if key not in data:
            if key in ('POINTING_AZIMUTH', 'LENS_ALTITUDE', 'RADIAL_DISTORTION'):
                continue  # optional for clients that predate camera pointing
            return None, 'Missing field: {0:s}'.format(key)
        try:
            if isinstance(data[key], bool):
                raise ValueError  # JSON booleans are not calibration numbers
            # json accepts literal Infinity/NaN; int(inf) raises OverflowError
            v = cast(float(data[key]))
        except (TypeError, ValueError, OverflowError):
            return None, 'Invalid value for {0:s}'.format(key)
        # Config accepts arbitrary finite latitude/longitude offsets.
        manual_offset = for_save and key in ('LATITUDE_OFFSET', 'LONGITUDE_OFFSET')
        if not math.isfinite(v) or (not manual_offset and not vmin <= v <= vmax):
            return None, '{0:s} out of range'.format(key)
        values[key] = v

    for key in ('PRECESSION', 'CALIBRATION_ENABLED'):
        if key in data:
            if not isinstance(data[key], bool):
                return None, '{0:s} must be a boolean'.format(key)
            values[key] = data[key]
    if for_save and 'CALIBRATION_ENABLED' in values:
        model = data.get('CALIBRATION')
        if model is not None:
            if not validateCalibration(model):
                return None, 'Invalid lens calibration; solve again'
            geometry = [values.get(k) for k in ('AZIMUTH_ANGLE', 'LATITUDE_OFFSET',
                'LONGITUDE_OFFSET', 'IMAGE_CIRCLE_DIAMETER', 'OFFSET_X', 'OFFSET_Y',
                'LENS_ALTITUDE', 'POINTING_AZIMUTH')]
            geometry += [values.get('RADIAL_DISTORTION', 0), int(values.get('PRECESSION', False))]
            # Older corrections belong to the original lens and catalogue convention.
            saved_geometry = model['geometry'] + ([0, 0] if model['version'] == 1 else [])
            if saved_geometry != geometry:
                return None, 'Alignment changed since calibration; solve again'
        # A successful solve may need no extra correction. Keep the opt-in
        # preference without blocking Save or applying an absent model.
        values['CALIBRATION'] = model
    return values, None


def applySolvedValuesToConfig(config, values):
    """Write overlay calibration and optional camera pointing, in place.
    The LENS_IMAGE_CIRCLE family drives unrelated behavior and stays unchanged.
    """
    config['LENS_AZIMUTH'] = values['AZIMUTH_ANGLE']
    if 'LENS_ALTITUDE' in values:
        config['LENS_ALTITUDE'] = values['LENS_ALTITUDE']

    if 'VIRTUALSKY' not in config:
        config['VIRTUALSKY'] = {}

    virtualsky = config['VIRTUALSKY']
    virtualsky['LATITUDE_OFFSET'] = values['LATITUDE_OFFSET']
    virtualsky['LONGITUDE_OFFSET'] = values['LONGITUDE_OFFSET']
    virtualsky['IMAGE_CIRCLE_DIAMETER'] = values['IMAGE_CIRCLE_DIAMETER']
    virtualsky['OFFSET_X'] = values['OFFSET_X']
    virtualsky['OFFSET_Y'] = values['OFFSET_Y']
    # Omitted extension fields leave existing settings intact for older clients.
    for key in ('POINTING_AZIMUTH', 'PRECESSION', 'RADIAL_DISTORTION'):
        if key in values:
            virtualsky[key] = values[key]
    if 'CALIBRATION_ENABLED' in values:
        virtualsky['CALIBRATION_ENABLED'] = values['CALIBRATION_ENABLED']
        virtualsky['CALIBRATION'] = values['CALIBRATION']

    config['LENS_SOLVED'] = True
    return config
