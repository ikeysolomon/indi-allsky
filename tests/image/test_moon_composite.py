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
    {'ROI': []}, {'ROI': [1, 0, 80, 80]},
    {'ROI': [80, 0, 0, 80]}, {'THRESHOLD': 255},
])
def test_invalid_configuration_is_explicit(settings):
    config = {'CAMERA_INTERFACE': 'indi', 'MOON_COMPOSITE': {
        'ENABLE': True, 'ROI': [0, 0, 80, 80], **settings,
    }}
    with pytest.raises(ValueError, match='MOON_COMPOSITE'):
        moon.validate_config(config)


def test_disabled_and_unsupported_configuration():
    moon.validate_config({})
    moon.validate_config({'MOON_COMPOSITE': {'ENABLE': False}})
    with pytest.raises(ValueError, match='native indi'):
        moon.validate_config({'CAMERA_INTERFACE': 'indi_passive', 'MOON_COMPOSITE': {'ENABLE': True}})


def capture_worker():
    cls = isolated_class('capture.py', 'CaptureWorker', {'shoot'}, {
        'logger': logging.getLogger('indi_allsky'),
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
def test_fits_roi_composite_reuses_normal_decoder(tmp_path, mode, bits):
    cls = isolated_class('processing.py', 'ImageProcessor', {
        'composite_moon', '_debayer', '__cfa_bgr_map', '__cfa_gray_map',
    }, {
        'time': time, 'numpy': numpy, 'cv2': cv2, 'math': math,
        'constants': constants,
        'logger': logging.getLogger('indi_allsky'),
        '__name__': 'indi_allsky.processing', '__package__': 'indi_allsky',
    })
    processor = cls()
    processor.config = {'MOON_COMPOSITE': {'ENABLE': True, 'ROI': [20, 20, 100, 100]}}
    processor.config['NIGHT_GRAYSCALE'] = mode == 'bayer_gray'
    processor.night_av = {constants.NIGHT_NIGHT: True, constants.NIGHT_MOONMODE: True}
    processor.max_bit_depth = bits
    processor._moon_display_lut = None
    dtype = numpy.uint8 if bits == 8 else numpy.uint16
    short = numpy.zeros((120, 120), dtype=dtype)
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
    processor.image = numpy.full((120, 120, 3) if color else (120, 120), 255, numpy.uint8)
    processor.composite_moon({
        'filename': str(filename), 'gain': 75, 'binning': 1,
        'camera_id': 7, 'exposure': 0.001,
    }, i_ref)
    assert numpy.all(processor.image[60, 57] == 90)
    assert numpy.all(processor.image[0, 0] == 255)
    assert not numpy.any(i_ref.hdulist[0].data)


@pytest.mark.parametrize('mismatch', ['gain', 'binning', 'camera_id', 'exposure'])
def test_mismatched_pair_is_logged_and_preserves_image(mismatch, caplog):
    cls = isolated_class('processing.py', 'ImageProcessor', {'composite_moon'}, {
        'time': time, 'logger': logging.getLogger('indi_allsky'), 'constants': constants,
        '__name__': 'indi_allsky.processing', '__package__': 'indi_allsky',
    })
    processor = cls()
    processor.config = {'MOON_COMPOSITE': {'ENABLE': True, 'ROI': [0, 0, 80, 80]}}
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
