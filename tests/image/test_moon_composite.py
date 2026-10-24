import ast
import io
import logging
import math
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
import tempfile
import time
from types import SimpleNamespace
from threading import RLock
from unittest import mock

from astropy.io import fits
import cv2
import numpy
import pytest

from indi_allsky import moon
from indi_allsky import constants
from indi_allsky import stretch as stretch_classes


ROOT = Path(__file__).resolve().parents[2]


def isolated_class(filename, name, methods, namespace):
    """Use real worker methods without Linux camera/Flask startup dependencies."""
    tree = ast.parse((ROOT / 'indi_allsky' / filename).read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    body = [
        node for node in original.body
        if (isinstance(node, ast.FunctionDef) and node.name in methods)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in methods for target in node.targets
        ))
    ]
    cls = ast.ClassDef(name=name, bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, filename, 'exec'), namespace)
    return namespace[name]


def lunar_patch(color=False):
    patch = numpy.full((80, 80), 3, numpy.uint8)
    cv2.circle(patch, (40, 40), 12, 180, -1)
    cv2.circle(patch, (37, 40), 4, 90, -1)
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) if color else patch


@pytest.mark.parametrize('margin', [0, 0.009, 0.101, True, '0.05', float('nan'), float('inf')])
def test_search_margin_is_validated(margin):
    with pytest.raises(ValueError, match='SEARCH_MARGIN'):
        moon.validate_config({
            'CAMERA_INTERFACE': 'indi',
            'MOON_COMPOSITE': {'ENABLE': True, 'EXPOSURE': 0.001, 'SEARCH_MARGIN': margin},
        })


@pytest.mark.parametrize('reported, matches', [
    (0.001, True), ('0.0010004', True), (0.001011, False),
    (None, False), (True, False), ('invalid', False), (0, False), (-0.001, False),
    ('nan', False), ('inf', False),
])
def test_driver_exposure_matching_is_positive_finite_and_bounded(reported, matches):
    assert moon.exposure_matches(reported, 0.001) is matches


def test_exptime_is_authoritative_and_alternate_exposure_does_not_hide_bad_metadata():
    assert moon.reported_exposure({'EXPOSURE': '0.001'}) == '0.001'
    assert moon.reported_exposure({'EXPTIME': 'invalid', 'EXPOSURE': 0.001}) == 'invalid'
    assert moon.reported_exposure({}) is None


def test_guided_selection_prefers_position_but_rejects_star_sized_sources():
    sensor = numpy.full((240, 320), 3, numpy.uint8)
    cv2.circle(sensor, (50, 120), 12, 220, -1)
    cv2.circle(sensor, (160, 120), 12, 150, -1)
    cv2.circle(sensor, (260, 120), 2, 240, -1)
    roi = moon.find_moon(sensor, expected=(160, 120), expected_radius=12)
    assert roi is not None and roi[0] < 160 < roi[2]
    star = sensor[:, 230:].copy()
    assert moon.find_moon(star, expected=(30, 120), expected_radius=12) is None


@pytest.mark.parametrize('phase', ['full', 'half', 'crescent'])
@pytest.mark.parametrize('radius, dilation, feather', [(12, 1, 1), (60, 1, 2), (180, 3, 4)])
def test_scaled_feather_preserves_lunar_detail_terminator_and_distant_sky(monkeypatch, phase, radius, dilation, feather):
    side = 4 * radius + 32
    center = side // 2
    yy, xx = numpy.ogrid[:side, :side]
    lit = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    if phase == 'half':
        lit &= xx >= center
    elif phase == 'crescent':
        lit &= (xx - (center - radius * 0.25)) ** 2 + (yy - center) ** 2 > radius ** 2
    patch = numpy.full((side, side), 3, numpy.uint8)
    patch[lit] = 160
    patch[lit & (xx % 9 < 4)] = 100
    image = numpy.full_like(patch, 25)
    image[lit] = 255
    original = image.copy()
    with mock.patch.object(cv2, 'GaussianBlur', wraps=cv2.GaussianBlur) as blur:
        applied = moon.blend(
            image, patch, (0, 0, side, side), expected=(center, center), expected_radius=radius,
        )
    if phase == 'crescent' and radius == 12:
        # A narrow low-resolution limb must fail safely, not become a filled disk.
        assert not applied
        numpy.testing.assert_array_equal(image, original)
        return
    assert applied
    assert blur.call_args.args[1] == (2 * feather + 1, 2 * feather + 1)
    interior = cv2.erode(lit.astype(numpy.uint8), numpy.ones((9, 9), numpy.uint8)).astype(bool)
    assert interior.any()
    numpy.testing.assert_array_equal(image[interior], patch[interior])
    reach = dilation + feather + 2
    outside = cv2.dilate(lit.astype(numpy.uint8), numpy.ones((2 * reach + 1, 2 * reach + 1), numpy.uint8)) == 0
    numpy.testing.assert_array_equal(image[outside], original[outside])


@pytest.mark.parametrize('radius, margin', [(12, 1), (180, 5)])
def test_feather_padding_rejects_near_edge_without_changing_image(radius, margin):
    side = 4 * radius + 32
    patch = numpy.full((side, side), 3, numpy.uint8)
    cv2.circle(patch, (radius + margin, side // 2), radius, 180, -1)
    image = numpy.full_like(patch, 25)
    original = image.copy()
    assert not moon.blend(image, patch, (0, 0, side, side))
    numpy.testing.assert_array_equal(image, original)


def test_underexposure_warning_keeps_manual_exposure_and_original_image(caplog):
    patch = numpy.full((80, 80), 3, numpy.uint8)
    cv2.circle(patch, (40, 40), 12, 6, -1)
    image = numpy.full_like(patch, 25)
    assert not moon.blend(image, patch, (0, 0, 80, 80))
    assert numpy.all(image == 25)
    assert 'too faint' in caplog.text and 'manual EXPOSURE' in caplog.text


@pytest.mark.parametrize('color', [False, True])
def test_surface_detail_and_untouched_sky(color):
    shape = (180, 220, 3) if color else (180, 220)
    image = numpy.full(shape, 25, numpy.uint8)
    cv2.circle(image, (100, 100), 14, (255, 255, 255) if color else 255, -1)
    before = image.copy()
    assert moon.blend(image, lunar_patch(color), [60, 60, 140, 140])
    assert numpy.all(image[100, 97] == 90)
    assert numpy.all(image[100, 105] == 180)
    assert numpy.array_equal(image[:60], before[:60])
    assert numpy.array_equal(image[:, :60], before[:, :60])
    assert numpy.array_equal(image[60:80], before[60:80])


@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('separate_display', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_buffered_blend_matches_original_arithmetic(monkeypatch, color, separate_display, seed):
    rng = numpy.random.default_rng(seed)
    shape = (120, 140, 3) if color else (120, 140)
    image = rng.integers(0, 256, size=shape, dtype=numpy.uint8)
    before = image.copy()
    patch = lunar_patch(color)
    display = rng.integers(0, 256, size=patch.shape, dtype=numpy.uint8) if separate_display else patch
    # Exercise strided display pixels as well as a strided target ROI.
    backing = numpy.empty((80, 160, 3) if color else (80, 160), numpy.uint8)
    backing[:, ::2] = display
    display = backing[:, ::2]
    original_display = display.copy()
    original_patch = patch.copy()
    gaussian_blur = cv2.GaussianBlur
    weights = []

    def record_alpha(*args, **kwargs):
        alpha = gaussian_blur(*args, **kwargs)
        weights.append(alpha.copy())
        return alpha

    monkeypatch.setattr(cv2, 'GaussianBlur', record_alpha)
    assert moon.blend(
        image, patch, [20, 20, 100, 100],
        display_patch=display if separate_display else None,
    )
    alpha = weights[0][:, :, None] if color else weights[0]
    expected = before.copy()
    pixels = display if separate_display else patch
    expected[20:100, 20:100] = numpy.rint(
        before[20:100, 20:100] * (1.0 - alpha) + pixels * alpha,
    ).astype(numpy.uint8)
    numpy.testing.assert_array_equal(image, expected)
    numpy.testing.assert_array_equal(patch, original_patch)
    numpy.testing.assert_array_equal(display, original_display)


def test_display_transfer_does_not_change_linear_mask_or_clipping_checks():
    patch = lunar_patch()
    display = (100 + patch // 2).astype(numpy.uint8)
    image = numpy.full((80, 80), 25, numpy.uint8)
    assert moon.blend(image, patch, [0, 0, 80, 80], display_patch=display)
    assert image[40, 40] == display[40, 40]
    assert image[10, 10] == 25
    assert moon.blend(image, patch, [0, 0, 80, 80], display_patch=numpy.full_like(patch, 255))
    clipped = patch.copy()
    clipped[patch >= 40] = 255
    before = image.copy()
    assert not moon.blend(image, clipped, [0, 0, 80, 80], display_patch=display)
    numpy.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('display', [
    numpy.zeros((2, 2), numpy.uint8),
    numpy.zeros((80, 80), numpy.uint16),
])
def test_invalid_display_patch_is_explicit(display):
    with pytest.raises(ValueError, match='display patch'):
        moon.blend(numpy.zeros((80, 80), numpy.uint8), lunar_patch(), [0, 0, 80, 80], display_patch=display)


@pytest.mark.parametrize('mode', stretch_classes.__all__)
@pytest.mark.parametrize('bits', [8, 12, 16])
@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('gamma', [0, 3])
def test_replayed_transfer_matches_sky_stretch_without_refitting(mode, bits, color, gamma):
    operator = getattr(stretch_classes, mode)(
        {'IMAGE_STRETCH': {'MODE1_GAMMA': gamma}}, mask={1: None},
    )
    shape = (80, 120, 3) if color else (80, 120)
    dtype = numpy.uint8 if bits == 8 else numpy.uint16
    data = numpy.random.default_rng(10).integers(0, 2 ** bits, size=shape, dtype=dtype)
    original = data.copy()
    operator.record_display_transfer = True
    stretched = operator.stretch(data, bits, 1)
    tables = operator._display_luts
    actual = operator.apply_last_stretch(data[20:60, 30:90])
    numpy.testing.assert_array_equal(actual, stretched[20:60, 30:90])
    numpy.testing.assert_array_equal(data, original)
    assert operator._display_luts is tables
    operator.record_display_transfer = False
    numpy.testing.assert_array_equal(operator.stretch(data, bits, 1), stretched)
    assert not operator._display_luts
    with pytest.raises(ValueError, match='No completed display stretch'):
        operator.apply_last_stretch(data[:2, :2])


@pytest.mark.parametrize('failure', ['empty', 'star', 'clipped', 'ambiguous', 'edge'])
def test_unsafe_patch_keeps_original(failure):
    patch = lunar_patch()
    if failure == 'empty':
        patch[:] = 0
    elif failure == 'star':
        patch[:] = 0
        patch[40, 40] = 200
    elif failure == 'clipped':
        patch[35:45, 35:45] = 255
    elif failure == 'ambiguous':
        cv2.circle(patch, (15, 15), 6, 160, -1)
    else:
        patch = numpy.roll(patch, 30, axis=1)
    image = numpy.full((80, 80), 255, numpy.uint8)
    before = image.copy()
    assert not moon.blend(image, patch, [0, 0, 80, 80])
    assert numpy.array_equal(image, before)


@pytest.mark.parametrize('settings', [
    {'EXPOSURE': 0}, {'EXPOSURE': float('nan')},
    {'EXPOSURE': float('inf')}, {'EXPOSURE': '0.001'}, {'EXPOSURE': True},
])
def test_invalid_configuration_is_explicit(settings):
    config = {'CAMERA_INTERFACE': 'indi', 'MOON_COMPOSITE': {
        'ENABLE': True, **settings,
    }}
    with pytest.raises(ValueError, match='MOON_COMPOSITE'):
        moon.validate_config(config)


def test_disabled_and_unsupported_configuration():
    moon.validate_config({})
    moon.validate_config({'MOON_COMPOSITE': {'ENABLE': False}})
    with pytest.raises(ValueError, match='native indi'):
        moon.validate_config({'CAMERA_INTERFACE': 'indi_passive', 'MOON_COMPOSITE': {'ENABLE': True}})


def test_legacy_manual_settings_do_not_constrain_detection():
    moon.validate_config({'CAMERA_INTERFACE': 'indi', 'MOON_COMPOSITE': {
        'ENABLE': True, 'EXPOSURE': 0.001, 'ROI': [0, 0, 0, 0], 'THRESHOLD': 255,
    }})


@pytest.mark.parametrize('radius', [5, 12, 40, 100])
@pytest.mark.parametrize('center', [(170, 180), (580, 400), (850, 700)])
@pytest.mark.parametrize('bayer', [False, True])
def test_detects_moving_moon_and_autoscales_even_roi(radius, center, bayer):
    sensor = numpy.full((950, 1100), 3, numpy.uint8)
    cv2.circle(sensor, center, radius, 180, -1)
    sensor[50, 50] = 255
    if bayer:
        sensor[::2, ::2] //= 2
    roi = moon.find_moon(sensor, bayer=bayer)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert all(value % 2 == 0 for value in roi)
    assert x1 <= center[0] - radius - 4
    assert y1 <= center[1] - radius - 4
    assert x2 >= center[0] + radius + 4
    assert y2 >= center[1] + radius + 4
    assert x2 - x1 < radius * 4 + 32
    assert y2 - y1 < radius * 4 + 32
    image = numpy.full_like(sensor, 25)
    assert moon.blend(image, sensor[y1:y2, x1:x2], roi)
    assert image[center[1], center[0]] == sensor[center[1], center[0]]
    assert image[0, 0] == 25


@pytest.mark.parametrize('brightness', [12, 30, 90, 230])
def test_threshold_adapts_to_short_exposure_brightness(brightness):
    sensor = numpy.full((200, 240), 2, numpy.uint8)
    cv2.circle(sensor, (140, 100), 15, brightness, -1)
    roi = moon.find_moon(sensor)
    assert roi is not None
    x1, y1, x2, y2 = roi
    image = numpy.full_like(sensor, 255)
    assert moon.blend(image, sensor[y1:y2, x1:x2], roi)
    assert image[100, 140] == brightness


@pytest.mark.parametrize('shape', ['empty', 'uniform', 'noise', 'star', 'line', 'rectangle', 'square', 'edge', 'ambiguous'])
def test_detection_rejects_unsafe_or_non_lunar_sources(shape):
    sensor = numpy.zeros((240, 320), numpy.uint8)
    if shape == 'uniform':
        sensor[:] = 120
    elif shape == 'noise':
        sensor[:] = numpy.random.default_rng(4).integers(0, 5, sensor.shape, dtype=numpy.uint8)
    elif shape == 'star':
        sensor[100, 120] = 220
    elif shape == 'line':
        cv2.line(sensor, (50, 50), (200, 200), 180, 5)
    elif shape in ('rectangle', 'square'):
        cv2.rectangle(sensor, (80, 80), (160 if shape == 'rectangle' else 120, 120), 180, -1)
    elif shape == 'edge':
        cv2.circle(sensor, (0, 100), 20, 180, -1)
    elif shape == 'ambiguous':
        cv2.circle(sensor, (80, 100), 15, 180, -1)
        cv2.circle(sensor, (220, 100), 15, 170, -1)
    assert moon.find_moon(sensor) is None


@pytest.mark.parametrize('distractors', [1, 2, 3])
def test_three_seed_budget_can_reject_two_brighter_non_circular_sources(distractors):
    sensor = numpy.zeros((320, 400), numpy.uint8)
    for index in range(distractors):
        x = 20 + index * 120
        cv2.rectangle(sensor, (x, 20), (x + 60, 80), 240 - index * 20, -1)
    cv2.circle(sensor, (280, 200), 20, 120, -1)
    with mock.patch.object(cv2, 'minMaxLoc', wraps=cv2.minMaxLoc) as seeds:
        roi = moon.find_moon(sensor)
    assert seeds.call_count == 3
    if distractors == 3:
        assert roi is None
        return
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 < 280 < x2 and y1 < 200 < y2


def test_hot_pixel_field_and_bright_crater_do_not_hide_lunar_extent():
    sensor = numpy.full((900, 1200), 3, numpy.uint8)
    rng = numpy.random.default_rng(12)
    sensor[rng.integers(0, 900, 300), rng.integers(0, 1200, 300)] = 255
    cv2.circle(sensor, (650, 450), 60, 75, -1)
    cv2.circle(sensor, (640, 440), 9, 230, -1)
    roi = moon.find_moon(sensor)
    assert roi is not None
    x1, y1, x2, y2 = roi
    assert x1 < 590 and y1 < 390 and x2 > 710 and y2 > 510
    image = numpy.full_like(sensor, 25)
    assert moon.blend(image, sensor[y1:y2, x1:x2], roi)
    assert image[450, 690] == 75
    assert image[440, 640] == 230


@pytest.mark.parametrize('layout', ['mono', 'rgb', 'bayer'])
def test_large_sensor_preview_and_decoder_roi_remain_bounded(layout):
    sensor = numpy.full((3010, 4010), 2, numpy.uint8)
    cv2.circle(sensor, (2801, 1701), 25, 160, -1)
    if layout == 'rgb':
        sensor = numpy.stack([sensor, sensor // 2, sensor // 3])
    with mock.patch.object(moon, '_select_region', wraps=moon._select_region) as select:
        roi = moon.find_moon(sensor, bayer=layout == 'bayer')
    assert roi is not None
    assert max(select.call_args.args[0].shape) <= 768
    x1, y1, x2, y2 = roi
    assert x1 <= 2772 and y1 <= 1672 and x2 >= 2830 and y2 >= 1730
    assert x2 - x1 < 130 and y2 - y1 < 130


@pytest.mark.parametrize('shape', [(31, 47), (769, 1027)])
@pytest.mark.parametrize('bayer', [False, True])
def test_preview_max_pooling_preserves_bright_seeds_and_sensor_coordinates(shape, bayer):
    sensor = numpy.random.default_rng(5).integers(0, 256, shape, dtype=numpy.uint8)
    step = max(1, math.ceil(max(shape) / 768))
    if bayer:
        step = max(2, step + step % 2)
    padded = numpy.pad(sensor, ((0, (-shape[0]) % step), (0, (-shape[1]) % step)))
    expected = padded.reshape(
        padded.shape[0] // step, step, padded.shape[1] // step, step,
    ).max(axis=(1, 3))
    with mock.patch.object(moon, '_select_region', return_value=None) as select:
        assert moon.find_moon(sensor, bayer=bayer) is None
    numpy.testing.assert_array_equal(select.call_args.args[0], expected)


@pytest.mark.parametrize('phase', [0.5, 0.75, 1.0])
def test_circular_limb_accepts_partial_lunar_disk(phase):
    sensor = numpy.full((240, 320), 3, numpy.uint8)
    cv2.circle(sensor, (160, 120), 30, 180, -1)
    if phase < 1:
        sensor[:, :int(160 - (phase - 0.5) * 60)] = 3
    assert moon.find_moon(sensor) is not None


def capture_worker():
    cls = isolated_class('capture.py', 'CaptureWorker', {'shoot'}, {
        'logger': logging.getLogger('indi_allsky'),
        '__name__': 'indi_allsky.capture', '__package__': 'indi_allsky',
    })
    worker = cls()
    worker.config = {'CAMERA_INTERFACE': 'indi', 'MOON_COMPOSITE': {'ENABLE': True, 'EXPOSURE': 0.001}}
    worker.indiclient = mock.Mock()
    worker.night = True
    worker.moonmode = True
    worker.focus_mode = False
    worker.moon_composite_suspended = False
    return worker


def test_capture_requests_short_frame_at_same_gain_and_binning():
    worker = capture_worker()
    worker.shoot(15.0, 75, 1, sync=False)
    worker.indiclient.beginMoonComposite.assert_called_once_with(15.0, 75, 1)
    worker.indiclient.setCcdExposure.assert_called_once_with(
        0.001, 75, 1, sync=False, timeout=None, sqm_exposure=False,
    )

    tree = ast.parse((ROOT / 'indi_allsky' / 'capture.py').read_text(encoding='utf-8'))
    waiting = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == 'waiting_for_frame'
    )
    # Execute the actual loop continuation, including its continue statement.
    continuation = ast.Module(body=[
        ast.For(
            target=ast.Name(id='_', ctx=ast.Store()),
            iter=ast.Tuple(elts=[ast.Constant(True)], ctx=ast.Load()),
            body=[waiting.body[0]], orelse=[],
        ),
    ], type_ignores=[])
    worker._shutdown = False
    scope = {'self': worker, 'now_time': 123.0}
    exec(compile(ast.fix_missing_locations(continuation), 'capture.py', 'exec'), scope)
    worker.indiclient.continueMoonComposite.assert_called_once_with()
    assert scope['frame_start_time'] == 123.0
    assert scope['camera_ready'] is False


@pytest.mark.parametrize('gate', ['day', 'moon_below', 'focus', 'sqm', 'sync', 'disabled', 'suspended'])
def test_capture_gates_preserve_normal_exposure(gate):
    worker = capture_worker()
    if gate == 'day':
        worker.night = False
    elif gate == 'moon_below':
        worker.moonmode = False
    elif gate == 'focus':
        worker.focus_mode = True
    elif gate == 'disabled':
        worker.config['MOON_COMPOSITE']['ENABLE'] = False
    elif gate == 'suspended':
        worker.moon_composite_suspended = True
    worker.shoot(15.0, 75, 1, sync=gate == 'sync', sqm_exposure=gate == 'sqm')
    worker.indiclient.beginMoonComposite.assert_not_called()
    assert worker.indiclient.setCcdExposure.call_args.args[0] == 15.0


def indi_client():
    cls = isolated_class(Path('camera') / 'indi.py', 'IndiClient', {
        'processBlob', '_processBlob', 'clearMoonComposite', 'getCcdExposureStatus',
    }, {
        'time': time, 'io': io, 'tempfile': tempfile, 'Path': Path, 'math': math,
        'datetime': datetime, 'logger': logging.getLogger('indi_allsky'),
        '__package__': 'indi_allsky.camera',
    })
    client = cls()
    client._moon_capture = moon.MoonCapture(1, (15.0, 75, 1), phase='short', status='waiting')
    client._moon_composite_lock = RLock()
    client.config = {'MOON_COMPOSITE': {'ENABLE': True}}
    client.exposureStartTime = time.time()
    client.exposure = 0.001
    client.gain = 75
    client.binning = 1
    client.sqm_exposure = False
    client.camera_id = 7
    client.indi_allsky_camera_name = 'test'
    client._filename_t = 'test'
    client.image_q = SimpleQueue()
    client._ctl_ccd_exposure = None
    client.ctl_ready = lambda _: (True, 'OK')
    return client


def test_pairing_waits_for_blobs_and_queues_one_job():
    client = indi_client()
    stream = io.BytesIO()
    fits.PrimaryHDU(lunar_patch(), header=fits.Header({'EXPTIME': 0.001})).writeto(stream)
    blob = SimpleNamespace(getblobdata=stream.getvalue)
    assert client.getCcdExposureStatus()[0] is False
    client.processBlob(blob)
    short = client._moon_capture.short_frame
    try:
        assert client.image_q.empty()
        assert client.getCcdExposureStatus()[0] is True
        client._moon_capture.phase = 'long'
        client.exposure = 15.0
        client._moon_capture.status = 'waiting'
        assert client.getCcdExposureStatus()[0] is False
        stream = io.BytesIO()
        fits.PrimaryHDU(lunar_patch(), header=fits.Header({'EXPTIME': 15.0})).writeto(stream)
        blob = SimpleNamespace(getblobdata=stream.getvalue)
        client.processBlob(blob)
        job = client.image_q.get_nowait()
        try:
            assert job['moon_composite'] == short
            assert job['exposure'] == 15.0
            assert job['moon_composite']['exposure'] == 0.001
            assert client.getCcdExposureStatus()[0] is True
            assert client._moon_capture.short_frame is None
            assert client._moon_capture.status == 'complete'
        finally:
            Path(job['filename']).unlink()
    finally:
        Path(short['filename']).unlink()


def test_first_bracket_cannot_overtake_a_previous_normal_blob():
    client = indi_client()
    client._moon_capture = moon.MoonCapture(0, status='waiting')
    assert client.getCcdExposureStatus()[0] is False
    client._moon_capture.status = 'complete'
    assert client.getCcdExposureStatus()[0] is True


def test_abandoned_pair_deletes_short_frame(tmp_path):
    client = indi_client()
    filename = tmp_path / 'moon.fit'
    filename.write_bytes(b'test')
    client._moon_capture.short_frame = {'filename': str(filename)}
    client.clearMoonComposite()
    assert not filename.exists()
    assert client._moon_capture.status == 'cancelled'
    assert client._moon_capture.short_frame is None


@pytest.mark.parametrize('fail', [False, True])
def test_worker_always_cleans_short_file(tmp_path, fail):
    cls = isolated_class('image.py', 'ImageWorker', {'processImage'}, {'Path': Path})
    worker = cls()
    filename = tmp_path / 'moon.fit'
    filename.write_bytes(b'test')
    worker._processImage = mock.Mock(side_effect=ValueError('bad long frame') if fail else None)
    job = {'moon_composite': {'filename': str(filename)}}
    if fail:
        with pytest.raises(ValueError):
            worker.processImage(job)
    else:
        worker.processImage(job)
    assert not filename.exists()


@pytest.mark.parametrize('mode', ['mono', 'rgb', 'bayer', 'bayer_gray'])
@pytest.mark.parametrize('bits', [8, 12, 16])
@pytest.mark.parametrize('sensor_size', [120, 1024])
@pytest.mark.parametrize('guided', [False, True])
def test_fits_roi_composite_reuses_normal_decoder(tmp_path, mode, bits, sensor_size, guided):
    cls = isolated_class('processing.py', 'ImageProcessor', {
        'composite_moon', '_debayer', '__cfa_bgr_map', '__cfa_gray_map',
    }, {
        'time': time, 'numpy': numpy, 'cv2': cv2, 'math': math,
        'constants': constants,
        'logger': logging.getLogger('indi_allsky'),
        '__name__': 'indi_allsky.processing', '__package__': 'indi_allsky',
    })
    processor = cls()
    processor.config = {'MOON_COMPOSITE': {'ENABLE': True}}
    processor.config['NIGHT_GRAYSCALE'] = mode == 'bayer_gray'
    processor.night_av = {constants.NIGHT_NIGHT: True, constants.NIGHT_MOONMODE: True}
    processor.position_av = {
        constants.POSITION_LATITUDE: 0, constants.POSITION_LONGITUDE: 0, constants.POSITION_ELEVATION: 0,
    }
    processor.max_bit_depth = bits
    processor._moon_display_lut = None
    dtype = numpy.uint8 if bits == 8 else numpy.uint16
    short = numpy.zeros((sensor_size, sensor_size), dtype=dtype)
    short[20:100, 20:100] = lunar_patch().astype(dtype) * (2 ** (bits - 8))
    if mode == 'rgb':
        short = numpy.stack([short, short, short])
    filename = tmp_path / 'short.fit'
    header = fits.Header({'EXPTIME': 0.001})
    if mode.startswith('bayer'):
        header['BAYERPAT'] = 'RGGB'
    fits.PrimaryHDU(short, header=header).writeto(filename)
    long_header = fits.PrimaryHDU(numpy.zeros_like(short), header=header).header
    long_header['EXPTIME'] = 15.0
    i_ref = SimpleNamespace(
        gain=75, binning=1, camera_id=7, exposure=15.0,
        hdulist=[SimpleNamespace(data=numpy.zeros_like(short), header=long_header)],
        image_bayerpat=header.get('BAYERPAT'),
    )
    color = mode in ('rgb', 'bayer')
    shape = (sensor_size, sensor_size, 3) if color else (sensor_size, sensor_size)
    processor.image = numpy.full(shape, 255, numpy.uint8)
    open_fits = fits.open
    opened = []
    read_sizes = []

    def open_short(*args, **kwargs):
        hdulist = open_fits(*args, **kwargs)
        opened.append(hdulist)
        readarray = hdulist._file.readarray

        def read_roi(*args, **kwargs):
            assert not hdulist._file.closed
            data = readarray(*args, **kwargs)
            read_sizes.append(data.nbytes)
            return data

        hdulist._file.readarray = read_roi
        return hdulist

    decoder = processor._debayer
    decoded_shapes = []

    def decode_roi(ref):
        decoded_shapes.append(ref.hdulist[0].data.shape[-2:])
        return decoder(ref)

    with mock.patch('astropy.io.fits.open', side_effect=open_short), mock.patch.object(processor, '_debayer', side_effect=decode_roi):
        guide = SimpleNamespace(
            search_region=mock.Mock(return_value=(20, 20, 100, 100)),
            expected_position=(60, 60), expected_radius=12,
        ) if guided else None
        processor.composite_moon({
            'filename': str(filename), 'gain': 75, 'binning': 1,
            'camera_id': 7, 'exposure': 0.001,
        }, i_ref, guide=guide)
    assert numpy.all(processor.image[60, 57] == 90)
    assert numpy.all(processor.image[0, 0] == 255)
    assert not numpy.any(i_ref.hdulist[0].data)
    expected_bytes = 80 * 80 * numpy.dtype(dtype).itemsize * (3 if mode == 'rgb' else 1) if guided else short.nbytes
    assert sum(read_sizes) == expected_bytes
    assert len(decoded_shapes) == 1
    assert all(dimension < 60 for dimension in decoded_shapes[0])
    assert len(opened) == 1
    assert opened[0][0]._data_loaded is not guided
    assert opened[0]._file.closed
    filename.unlink()
    assert not filename.exists()


@pytest.mark.parametrize('mismatch', ['gain', 'binning', 'camera_id', 'exposure'])
def test_mismatched_pair_is_logged_and_preserves_image(mismatch, caplog):
    cls = isolated_class('processing.py', 'ImageProcessor', {'composite_moon'}, {
        'time': time, 'logger': logging.getLogger('indi_allsky'), 'constants': constants,
        '__name__': 'indi_allsky.processing', '__package__': 'indi_allsky',
    })
    processor = cls()
    processor.config = {'MOON_COMPOSITE': {'ENABLE': True}}
    processor.night_av = {constants.NIGHT_NIGHT: True, constants.NIGHT_MOONMODE: True}
    processor.image = numpy.full((80, 80), 255, numpy.uint8)
    before = processor.image.copy()
    frame = {'gain': 75, 'binning': 1, 'camera_id': 7, 'exposure': 0.001}
    frame[mismatch] = 99
    processor.composite_moon(frame, SimpleNamespace(gain=75, binning=1, camera_id=7, exposure=15))
    assert numpy.array_equal(before, processor.image)
    assert 'Moon composite skipped:' in caplog.text


@pytest.mark.parametrize('night,moonmode,enabled', [
    (False, False, True), (False, True, True), (True, False, True), (True, True, False),
])
def test_inactive_moon_mode_performs_no_short_frame_io(night, moonmode, enabled):
    cls = isolated_class('processing.py', 'ImageProcessor', {'composite_moon'}, {
        'time': time, 'constants': constants,
        '__name__': 'indi_allsky.processing', '__package__': 'indi_allsky',
    })
    processor = cls()
    processor.config = {'MOON_COMPOSITE': {'ENABLE': enabled}}
    processor.night_av = {constants.NIGHT_NIGHT: night, constants.NIGHT_MOONMODE: moonmode}
    with mock.patch('astropy.io.fits.open') as read_short, mock.patch.object(moon, 'blend') as blend:
        processor.composite_moon({}, None)
    read_short.assert_not_called()
    blend.assert_not_called()


@pytest.mark.parametrize('night,moonmode,enabled,focus,sqm,sync,waiting', [
    (True, True, True, False, False, False, True),
    (True, False, True, False, False, False, False),
    (False, False, True, False, False, False, False),
    (False, True, True, False, False, False, False),
    (True, True, False, False, False, False, False),
    (True, True, True, True, False, False, False),
    (True, True, True, False, True, False, False),
    (True, True, True, False, False, True, False),
])
def test_extra_blob_waits_only_in_active_moon_mode(night, moonmode, enabled, focus, sqm, sync, waiting):
    cls = isolated_class(Path('camera') / 'indi.py', 'IndiClient', {'setCcdExposure'}, {
        'time': time, 'constants': constants, 'MoonCapture': moon.MoonCapture,
    })
    client = cls()
    client._moon_composite_lock = RLock()
    client._moon_capture = None
    client.config = {'MOON_COMPOSITE': {'ENABLE': enabled}, 'FOCUS_MODE': focus}
    client.night_av = {constants.NIGHT_NIGHT: night, constants.NIGHT_MOONMODE: moonmode}
    client.gain, client.binning, client.timeout = 75, 1, 10
    client.ccd_device = None
    client.set_number = mock.Mock()
    client._expUtils = SimpleNamespace()
    client.setCcdExposure(15, 75, 1, sync=sync, sqm_exposure=sqm)
    assert (client._moon_capture is not None and client._moon_capture.status == 'waiting') is waiting
