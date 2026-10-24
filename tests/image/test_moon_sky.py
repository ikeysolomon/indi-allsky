import copy
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import cv2
import numpy
import pytest

from indi_allsky import constants, moon, moon_sky
from indi_allsky.exceptions import ConfigSaveException
from indi_allsky.lens_solver import applySolvedValuesToConfig
from indi_allsky.lens_solver.calibration import pipelineSignature
from indi_allsky.moon_sky import MoonSkyGuide, image_geometry
from test_moon_composite import isolated_class


def config():
    return {
        'IMAGE_SCALE': 100, 'LENS_AZIMUTH': 0.0, 'LENS_ALTITUDE': 90.0,
        'MOON_COMPOSITE': {'ENABLE': True, 'EXPOSURE': 0.001},
        'VIRTUALSKY': {},
    }


def reference(shape=(240, 320), binning=1):
    return SimpleNamespace(
        camera_uuid='camera-1', binning=binning,
        hdulist=[SimpleNamespace(data=numpy.zeros(shape, numpy.uint8), header={})],
        exposure=0.1, exp_elapsed=0.1,
        exp_date=datetime(2026, 10, 24, 10, tzinfo=timezone.utc),
    )


def manual(cfg):
    cfg['VIRTUALSKY']['ALIGNMENT'] = {'source': 'manual', 'pipeline': pipelineSignature(cfg)}


def solved():
    return {
        'success': True, 'partial': False, 'message': 'Matched 100 stars',
        'quality': {'rms_px': 0.5, 'stars_matched': 100},
        'values': {
            'AZIMUTH_ANGLE': 25.0, 'LENS_ALTITUDE': 90.0, 'POINTING_AZIMUTH': 0.0,
            'LATITUDE_OFFSET': 0.0, 'LONGITUDE_OFFSET': 0.0,
            'IMAGE_CIRCLE_DIAMETER': 220, 'OFFSET_X': 0, 'OFFSET_Y': 0,
            'RADIAL_DISTORTION': 0.0, 'PRECESSION': True,
        },
    }


@pytest.mark.parametrize('changes', [
    {}, {'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE'},
    {'IMAGE_ROTATE': 'ROTATE_90_COUNTERCLOCKWISE'}, {'IMAGE_ROTATE': 'ROTATE_180'},
    {'IMAGE_FLIP_H': True}, {'IMAGE_FLIP_V': True},
    {'IMAGE_ROTATE_ANGLE': 19}, {'IMAGE_ROTATE_ANGLE': -27, 'IMAGE_ROTATE_KEEP_SIZE': True},
    {'IMAGE_CROP_ROI': [20, 30, 540, 400]},
    {'IMAGE_CROP_IMAGE_CIRCLE': True, 'LENS_IMAGE_CIRCLE': 180, 'LENS_OFFSET_X': 12, 'LENS_OFFSET_Y': -8},
    {'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE', 'IMAGE_ROTATE_ANGLE': 19,
     'IMAGE_FLIP_H': True, 'IMAGE_FLIP_V': True, 'IMAGE_SCALE': 73,
     'IMAGE_BORDER': {'TOP': 8, 'LEFT': 12, 'RIGHT': 4, 'BOTTOM': 6}},
])
def test_sensor_to_overlay_matrix_matches_real_image_pipeline(changes):
    cfg = {**config(), **changes}
    cls = isolated_class('processing.py', 'ImageProcessor', {
        'rotate_90', '_rotate_90', 'rotate_angle', '_rotate_angle', 'flip_v', 'flip_h', '_flip',
        'crop_image', '_crop_image', 'scale_image', '_scale_image', 'add_border', '_add_border',
    }, {'numpy': numpy, 'cv2': cv2, 'logger': logging.getLogger('indi_allsky')})
    processor = cls()
    processor.config = cfg
    processor.focus_mode = False
    processor.getLatestImage = lambda: reference(binning=2)
    processor.image = numpy.zeros((240, 320, 3), numpy.uint8)
    cv2.circle(processor.image, (160, 120), 4, (255, 255, 255), -1)
    matrix, shape = image_geometry(cfg, (240, 320), 2)
    processor.rotate_90()
    processor.rotate_angle()
    processor.flip_v()
    processor.flip_h()
    processor.crop_image()
    processor.scale_image()
    processor.add_border()
    assert processor.image.shape[:2] == shape
    weights = processor.image[:, :, 0].astype(float)
    yy, xx = numpy.indices(shape)
    actual = [(xx * weights).sum() / weights.sum(), (yy * weights).sum() / weights.sum()]
    expected = (matrix @ [160, 120, 1])[:2]
    numpy.testing.assert_allclose(actual, expected, atol=0.5)
    numpy.testing.assert_allclose(numpy.linalg.solve(matrix, [*expected, 1]), [160, 120, 1], atol=1e-8)


def test_first_image_solve_is_exact_persistent_and_not_repeated(monkeypatch):
    cfg, ref = config(), reference()
    guide = MoonSkyGuide(cfg)
    inputs = []

    def solve(self, filename, latitude, longitude, timestamp, initial, **kwargs):
        assert Path(filename).is_file()
        inputs.append(Path(filename))
        assert timestamp == pytest.approx(ref.exp_date.timestamp() - 0.05)
        assert kwargs['sensor_shape'] == (240, 320)
        assert initial['CALIBRATION_ENABLED'] and initial['PRECESSION']
        numpy.testing.assert_array_equal(cv2.imread(str(filename), 0), numpy.full((240, 320), 25, numpy.uint8))
        return solved()

    monkeypatch.setattr(moon_sky.IndiAllSkyLensSolver, 'solve', solve)
    persist = mock.Mock(return_value=True)
    guide.align_first_image(numpy.full((240, 320), 25, numpy.uint8), ref, 40.1, -75.4, persist)
    persist.assert_called_once()
    assert not inputs[0].exists()
    values, saved = persist.call_args.args
    assert saved['source'] == 'automatic'
    assert saved['camera_uuid'] == ref.camera_uuid
    assert saved['image_size'] == [320, 240]
    assert cfg['LENS_AZIMUTH'] == 25
    restarted = MoonSkyGuide(copy.deepcopy(cfg))
    assert restarted.is_aligned(ref)
    restarted.align_first_image(numpy.zeros((240, 320), numpy.uint8), ref, 40.1, -75.4, persist)
    assert len(inputs) == 1


@pytest.mark.parametrize('failure', ['cloud', 'partial', 'save'])
def test_failed_alignment_keeps_config_unchanged_and_backs_off(monkeypatch, caplog, failure):
    cfg, ref = config(), reference()
    original = copy.deepcopy(cfg)
    result = solved()
    if failure == 'cloud':
        result.update(success=False, message='Only 0 stars detected')
    elif failure == 'partial':
        result.update(partial=True, message='Incomplete pointing')
    solve = mock.Mock(return_value=result)
    monkeypatch.setattr(moon_sky.IndiAllSkyLensSolver, 'solve', solve)
    persist = mock.Mock(return_value=False)
    guide = MoonSkyGuide(cfg)
    for _ in range(2):
        guide.align_first_image(numpy.zeros((240, 320), numpy.uint8), ref, 40.1, -75.4, persist)
    solve.assert_called_once()
    assert persist.call_count == (1 if failure == 'save' else 0)
    assert cfg == original
    assert 'failed' in caplog.text or 'not saved' in caplog.text
    assert not guide.is_aligned(ref)


def test_actual_cloudy_first_image_reports_solver_failure(caplog):
    cfg = config()
    guide = MoonSkyGuide(cfg)
    persist = mock.Mock()
    guide.align_first_image(numpy.full((240, 320), 10, numpy.uint8), reference(), 40.1, -75.4, persist)
    persist.assert_not_called()
    assert 'Automatic Moon alignment failed' in caplog.text
    assert 'stars detected' in caplog.text


def test_actual_first_image_star_solve_and_restart_reuse():
    from indi_allsky.lens_solver import predictAltAz, projectToPixels
    from indi_allsky.lens_solver.projection import precessCatalog

    cfg = {**config(), 'LENS_IMAGE_CIRCLE': 1700}
    ref = reference((1920, 1920))
    guide = MoonSkyGuide(cfg)
    stars = moon_sky.IndiAllSkyLensSolver(cfg).loadCatalog()
    timestamp = ref.exp_date.timestamp() - 0.05
    stars = precessCatalog(stars, timestamp)
    alt, az = predictAltAz(stars, 40.1, -75.4, timestamp)
    keep = alt > numpy.radians(10)
    x, y = projectToPixels(alt[keep], az[keep], [15, 0, 0, 1700, 0, 0], 1920, 1920)
    image = numpy.full((1920, 1920), 10, numpy.uint8)
    for sx, sy in zip(x, y):
        if 5 < sx < 1915 and 5 < sy < 1915:
            cv2.circle(image, (round(sx), round(sy)), 2, 220, -1)
    image = cv2.GaussianBlur(image, (5, 5), 1.1)
    persist = mock.Mock(return_value=True)
    guide.align_first_image(image, ref, 40.1, -75.4, persist)
    persist.assert_called_once()
    assert abs(cfg['LENS_AZIMUTH'] - 15) < 1
    assert abs(cfg['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] - 1700) < 20
    assert cfg['VIRTUALSKY']['ALIGNMENT']['quality']['stars_matched'] >= 30
    assert MoonSkyGuide(copy.deepcopy(cfg)).is_aligned(ref)


@pytest.mark.parametrize('kind', ['recorded', 'unrecorded', 'stale'])
def test_only_explicit_current_alignment_is_reused(monkeypatch, kind):
    cfg = config()
    if kind == 'unrecorded':
        cfg['VIRTUALSKY']['OFFSET_X'] = 15
    else:
        manual(cfg)
    if kind == 'stale':
        cfg['IMAGE_FLIP_H'] = True
    solve = mock.Mock(return_value=solved())
    monkeypatch.setattr(moon_sky.IndiAllSkyLensSolver, 'solve', solve)
    MoonSkyGuide(cfg).align_first_image(numpy.zeros((240, 320), numpy.uint8), reference(), 40.1, -75.4, mock.Mock())
    assert solve.call_count == (0 if kind == 'recorded' else 1)


@pytest.mark.parametrize('date_obs', ['2026-10-24T10:00:00Z', '2026-10-24T10:00:00', '2026-10-24T20:00:00+10:00', None])
def test_guidance_uses_short_exposure_time_not_processing_time(monkeypatch, date_obs):
    cfg, ref = config(), reference()
    cfg['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 220
    manual(cfg)
    observed = []

    def moon_at(obs):
        observed.append(obs.date.datetime().replace(tzinfo=timezone.utc).timestamp())
        return SimpleNamespace(alt=math.pi / 2, az=0)

    monkeypatch.setattr(moon_sky.ephem, 'Moon', moon_at)
    frame = {'exposure': 0.002, 'exp_time': ref.exp_date.timestamp() + 0.002}
    header = {'DATE-OBS': date_obs} if date_obs else {}
    bounds = MoonSkyGuide(cfg).search_region(frame, ref, header, 40.1, -75.4, 100)
    assert observed[0] == pytest.approx(ref.exp_date.timestamp() + 0.001, abs=1e-4)
    assert bounds[0] < 160 < bounds[2] and bounds[1] < 120 < bounds[3]
    assert all(value % 2 == 0 for value in bounds)


def test_guided_search_ignores_brighter_circular_streetlights_elsewhere(monkeypatch):
    cfg, ref = config(), reference((800, 1000))
    cfg['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 700
    manual(cfg)
    monkeypatch.setattr(moon_sky.ephem, 'Moon', lambda obs: SimpleNamespace(alt=math.pi / 2, az=0))
    search = MoonSkyGuide(cfg).search_region(
        {'exposure': 0.001, 'exp_time': ref.exp_date.timestamp()}, ref, {}, 40.1, -75.4, 0,
    )
    sensor = numpy.full((800, 1000), 3, numpy.uint8)
    cv2.circle(sensor, (100, 100), 30, 255, -1)
    cv2.circle(sensor, (850, 600), 25, 240, -1)
    cv2.circle(sensor, (500, 400), 12, 100, -1)
    roi = moon.find_moon(sensor, search=search, bayer=True)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 < 500 < x2 and y1 < 400 < y2
    image = numpy.full_like(sensor, 25)
    assert moon.blend(image, sensor[y1:y2, x1:x2], roi)
    assert image[400, 500] == 100
    assert image[100, 100] == 25


def test_saved_reference_binning_and_pipeline_are_inverted(monkeypatch):
    cfg = {**config(), 'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE',
           'IMAGE_FLIP_H': True, 'IMAGE_SCALE': 73, 'IMAGE_BORDER': {'TOP': 8, 'LEFT': 12}}
    ref = reference((240, 320))
    matrix, (height, width) = image_geometry(cfg, (240, 320), 1)
    target = matrix @ [180, 100, 1]
    cfg['VIRTUALSKY'].update(
        IMAGE_CIRCLE_DIAMETER=150, OFFSET_X=target[0] - width / 2,
        OFFSET_Y=height / 2 - target[1],
        ALIGNMENT={
            'source': 'automatic', 'pipeline': pipelineSignature(cfg), 'camera_uuid': ref.camera_uuid,
            'binning': 1, 'sensor_shape': [240, 320], 'image_size': [width, height],
            'latitude': 40.1, 'longitude': -75.4,
        },
    )
    monkeypatch.setattr(moon_sky.ephem, 'Moon', lambda obs: SimpleNamespace(alt=math.pi / 2, az=0))
    current = reference((120, 160), binning=2)
    bounds = MoonSkyGuide(cfg).search_region(
        {'exposure': 0.001, 'exp_time': ref.exp_date.timestamp()}, current, {}, 40.1, -75.4, 0,
    )
    assert bounds[0] < 90 < bounds[2] and bounds[1] < 50 < bounds[3]
    assert abs((bounds[0] + bounds[2]) / 2 - 89.75) < 2
    assert abs((bounds[1] + bounds[3]) / 2 - 49.75) < 2


@pytest.mark.parametrize('failure', ['below-horizon', 'outside', 'missing-time', 'invalid-time', 'stale'])
def test_bad_predictions_are_explicit_and_never_fall_back_to_whole_frame(monkeypatch, failure):
    cfg, ref = config(), reference()
    cfg['VIRTUALSKY']['IMAGE_CIRCLE_DIAMETER'] = 220
    manual(cfg)
    altitude = -0.1 if failure == 'below-horizon' else math.pi / 2
    monkeypatch.setattr(moon_sky.ephem, 'Moon', lambda obs: SimpleNamespace(alt=altitude, az=0))
    if failure == 'outside':
        cfg['VIRTUALSKY']['OFFSET_X'] = 1000
    if failure == 'stale':
        cfg['IMAGE_FLIP_H'] = True
    frame = {'exposure': 0.001, 'exp_time': ref.exp_date.timestamp()}
    if failure == 'missing-time':
        del frame['exp_time']
    if failure == 'invalid-time':
        frame['exp_time'] = float('nan')
    with pytest.raises(ValueError):
        MoonSkyGuide(cfg).search_region(frame, ref, {}, 40.1, -75.4, 0)


@pytest.mark.parametrize('failure', [None, 'concurrent', 'save'])
def test_worker_persists_latest_config_and_camera_metadata_atomically(monkeypatch, caplog, failure):
    cfg = config()
    latest = SimpleNamespace(config=copy.deepcopy(cfg), save=mock.Mock())
    if failure == 'concurrent':
        latest.config['VIRTUALSKY']['OFFSET_X'] = 30
    if failure == 'save':
        latest.config = cfg.copy()
        latest.save.side_effect = ConfigSaveException('database rejected configuration')
    module = ModuleType('indi_allsky.config')
    module.IndiAllSkyConfig = lambda: latest
    monkeypatch.setitem(sys.modules, module.__name__, module)
    database = SimpleNamespace(session=mock.Mock())
    cls = isolated_class('image.py', 'ImageWorker', {'save_moon_sky_alignment'}, {
        '__name__': 'indi_allsky.image', '__package__': 'indi_allsky',
        'logger': logging.getLogger('indi_allsky'), 'db': database,
    })
    worker = cls()
    worker.config = cfg
    camera = SimpleNamespace(az=0, alt=90, data={'unrelated': 'retained'})
    values = {**solved()['values'], 'CALIBRATION_ENABLED': True, 'CALIBRATION': None}
    record = {'source': 'automatic'}
    assert worker.save_moon_sky_alignment(values, record, camera) is (failure is None)
    if failure == 'concurrent':
        latest.save.assert_not_called()
        assert worker.config['VIRTUALSKY']['OFFSET_X'] == 30
    elif failure == 'save':
        database.session.rollback.assert_called_once()
        assert 'Unable to persist' in caplog.text
        assert cfg == config()
    else:
        latest.save.assert_called_once_with('system', 'Automatic first-image VirtualSky alignment for Moon recovery')
        assert latest.config['VIRTUALSKY']['ALIGNMENT'] == record
        assert camera.data['vs_image_circle_diameter'] == 220
        assert camera.data['unrelated'] == 'retained'
        assert camera.az == 25


def test_worker_alignment_is_gated_outside_enabled_night_capture():
    cls = isolated_class('image.py', 'ImageWorker', {'moon_sky_guide'}, {
        '__name__': 'indi_allsky.image', '__package__': 'indi_allsky', 'constants': constants,
    })
    worker = cls()
    worker.config = config()
    worker.night_av = {constants.NIGHT_NIGHT: False}
    assert worker.moon_sky_guide() is None
    worker.night_av[constants.NIGHT_NIGHT] = True
    worker.config['FOCUS_MODE'] = True
    assert worker.moon_sky_guide() is None
    worker.config['FOCUS_MODE'] = False
    worker.config['MOON_COMPOSITE']['ENABLE'] = False
    assert worker.moon_sky_guide() is None
