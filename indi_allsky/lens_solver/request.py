# (key, cast, min, max) -- ranges match the config form validators.
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
    'LENS_OFFSET_X',
    'LENS_OFFSET_Y',
    'LENS_IMAGE_CIRCLE',
    'IMAGE_ROTATE',
    'IMAGE_ROTATE_ANGLE',
    'IMAGE_ROTATE_KEEP_SIZE',
    'IMAGE_FLIP_V',
    'IMAGE_FLIP_H',
)

# the VIRTUALSKY sub-keys the solver itself writes
LENS_GEOMETRY_VIRTUALSKY_KEYS = (
    'IMAGE_CIRCLE_DIAMETER',
    'LATITUDE_OFFSET',
    'LONGITUDE_OFFSET',
    'OFFSET_X',
    'OFFSET_Y',
)


def captureLensGeometrySnapshot(config):
    """Snapshot every config value that affects the solved pixel geometry,
    to be compared later via ``invalidateLensSolveIfGeometryChanged``.
    """
    virtualsky = config.get('VIRTUALSKY', {})
    return (
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


def parseSolverRequestValues(data):
    """Validate and coerce the six solver form values from request JSON.
    Returns (values, None) or (None, error); only the six known keys are
    ever passed through.
    """
    values = {}
    for key, cast, vmin, vmax in SOLVER_REQUEST_FIELDS:
        if key not in data:
            return None, 'Missing field: {0:s}'.format(key)
        try:
            # json accepts literal Infinity/NaN; int(inf) raises OverflowError
            v = cast(float(data[key]))
        except (TypeError, ValueError, OverflowError):
            return None, 'Invalid value for {0:s}'.format(key)
        # NaN comparisons are always False, so this also rejects NaN
        if not vmin <= v <= vmax:
            return None, '{0:s} out of range'.format(key)
        values[key] = v

    return values, None


def applySolvedValuesToConfig(config, values):
    """Write exactly LENS_AZIMUTH, the five VIRTUALSKY offset/diameter keys,
    and LENS_SOLVED, in place -- never LENS_ALTITUDE or the LENS_IMAGE_CIRCLE
    family, which drive unrelated behavior. LENS_SOLVED is the sole gate the
    Milky Way enhancement trusts to know the geometry is real, so it must
    only ever be set here, never defaulted True.
    """
    config['LENS_AZIMUTH'] = values['AZIMUTH_ANGLE']
    config['LENS_SOLVED'] = True

    if 'VIRTUALSKY' not in config:
        config['VIRTUALSKY'] = {}

    virtualsky = config['VIRTUALSKY']
    virtualsky['LATITUDE_OFFSET'] = values['LATITUDE_OFFSET']
    virtualsky['LONGITUDE_OFFSET'] = values['LONGITUDE_OFFSET']
    virtualsky['IMAGE_CIRCLE_DIAMETER'] = values['IMAGE_CIRCLE_DIAMETER']
    virtualsky['OFFSET_X'] = values['OFFSET_X']
    virtualsky['OFFSET_Y'] = values['OFFSET_Y']

    return config
