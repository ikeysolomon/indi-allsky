"""Imported capture/camera/FITS boundaries with simulated external services.

These tests run real constructors and the capture loop, not AST-extracted methods.
The INDI transport and application database are fakes, not hardware evidence.
"""

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import heapq
import importlib.metadata
import importlib.util
import io
import queue
from multiprocessing import Array, Queue
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest import mock

from astropy.io import fits
from flask import Flask
import numpy
import pytest

from indi_allsky import constants


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky'


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    pyindi = ModuleType('PyIndi')
    for index, name in enumerate((
        'IPS_IDLE', 'IPS_OK', 'IPS_BUSY', 'IPS_ALERT', 'ISS_OFF', 'ISS_ON',
        'ISR_1OFMANY', 'ISR_ATMOST1', 'ISR_NOFMANY', 'INDI_NUMBER', 'INDI_SWITCH',
        'INDI_TEXT', 'INDI_LIGHT', 'INDI_BLOB', 'INDI_UNKNOWN', 'IP_RO', 'IP_WO', 'IP_RW', 'B_ALSO',
    )):
        setattr(pyindi, name, index)
    pyindi.BaseClient = type('BaseClient', (), {'disconnectServer': lambda self: setattr(self, 'disconnected', True)})
    pyindi.BaseMediator = type('BaseMediator', (), {})
    pyindi.BaseDevice = SimpleNamespace(**{
        name + '_INTERFACE': index for index, name in enumerate((
            'GENERAL', 'TELESCOPE', 'CCD', 'GUIDER', 'FOCUSER', 'FILTER', 'DOME',
            'GPS', 'WEATHER', 'AO', 'DUSTCAP', 'LIGHTBOX', 'DETECTOR', 'ROTATOR', 'AUX',
        ))
    })
    pyindi.PropertyBlob = lambda prop: prop
    monkeypatch.setitem(sys.modules, 'PyIndi', pyindi)
    monkeypatch.setitem(sys.modules, 'dbus', ModuleType('dbus'))
    monkeypatch.setitem(sys.modules, 'astroalign', ModuleType('astroalign'))

    app = Flask('moon-integration')
    flask_module = ModuleType('indi_allsky.flask')
    flask_module.__path__ = [str(ROOT / 'flask')]
    flask_module.create_app = lambda: app
    flask_module.db = mock.Mock(name='external_database')
    models = ModuleType('indi_allsky.flask.models')
    for name in (
        'TaskQueueQueue', 'TaskQueueState', 'IndiAllSkyDbCameraTable',
        'IndiAllSkyDbImageTable', 'IndiAllSkyDbTaskQueueTable',
        'IndiAllSkyDbBadPixelMapTable', 'IndiAllSkyDbDarkFrameTable',
        'IndiAllSkyDbTleDataTable', 'IndiAllSkyDbFitsImageTable', 'NotificationCategory',
    ):
        setattr(models, name, mock.Mock(name=name))
    database = ModuleType('indi_allsky.flask.miscDb')
    database.miscDb = mock.Mock(name='external_miscDb')
    for module in (flask_module, models, database):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    camera_package = ModuleType('indi_allsky.camera')
    camera_package.__path__ = [str(ROOT / 'camera')]
    monkeypatch.setitem(sys.modules, camera_package.__name__, camera_package)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name:
                        'test-transport' if name == 'pyindi-client' else original_version(name))
    camera = load_module(monkeypatch, 'indi_allsky.camera._moon_test_indi', ROOT / 'camera' / 'indi.py')
    camera_package.indi = camera.IndiClient
    capture = load_module(monkeypatch, 'indi_allsky._moon_test_capture', ROOT / 'capture.py')
    processing = load_module(monkeypatch, 'indi_allsky._moon_test_processing', ROOT / 'processing.py')
    monkeypatch.setattr(camera, 'tempfile', SimpleNamespace(
        NamedTemporaryFile=partial(tempfile.NamedTemporaryFile, dir=tmp_path),
    ))

    config = {
        'CAMERA_INTERFACE': 'indi', 'IMAGE_FOLDER': str(tmp_path), 'VARLIB_FOLDER': str(tmp_path),
        'MOON_COMPOSITE': {'ENABLE': True, 'EXPOSURE': 0.001, 'ROI': [0, 0, 80, 80], 'THRESHOLD': 40},
        'NIGHT_SUN_ALT_DEG': -6, 'NIGHT_MOONMODE_ALT_DEG': 0,
        'NIGHT_MOONMODE_PHASE': 33, 'EXPOSURE_PERIOD': 1.0, 'EXPOSURE_PERIOD_DAY': 1.0,
        'IMAGE_FLIP_V': False, 'KEOGRAM_ANGLE': 0, 'TARGET_ADU': 75, 'TARGET_ADU_DAY': 75,
        'ORB_PROPERTIES': {'SUN_COLOR': [255, 255, 255], 'MOON_COLOR': [255, 255, 255]},
    }
    shared = SimpleNamespace(
        position=Array('d', [0] * 5), exposure=Array('i', [0] * 7),
        gain=Array('i', [0] * 10), binning=Array('i', [1] * 6),
        night=Array('i', [1, 1]), sensors_temp=Array('d', [0] * 64),
        sensors_user=Array('d', [0] * 64), astro=Array('d', [0] * 64),
    )
    queues = [Queue() for _ in range(5)]
    error_q, capture_q, image_q, video_q, upload_q = queues
    client = camera.IndiClient(config, image_q, shared.position, shared.exposure,
                              shared.gain, shared.binning, shared.night)
    client.gain, client.binning, client.camera_id = 75, 1, 7
    client.ccd_device = SimpleNamespace(getDeviceName=lambda: 'test-camera')
    client._indi_allsky_camera_name = 'test-camera'
    client.disableCcdCooler = mock.Mock()
    client.abortCcdExposure = mock.Mock()
    ready = SimpleNamespace(value=True)
    control = SimpleNamespace(getState=lambda: pyindi.IPS_OK if ready.value else pyindi.IPS_BUSY)
    client.set_number = mock.Mock(return_value=control)
    worker = capture.CaptureWorker(
        1, config, error_q, capture_q, image_q, video_q, upload_q,
        shared.position, shared.exposure, shared.gain, shared.binning,
        shared.sensors_temp, shared.sensors_user, shared.night, shared.astro,
    )
    worker.indiclient = client
    worker.night = worker.moonmode = True
    worker._expUtils.EXPOSURE_NEXT = 0.1
    worker._expUtils.GAIN_NEXT = 75
    worker._expUtils.BINNING_NEXT = 1
    processor = processing.ImageProcessor(
        config, shared.position, shared.exposure, shared.gain, shared.binning,
        shared.sensors_temp, shared.sensors_user, shared.night, shared.astro,
    )
    try:
        yield SimpleNamespace(camera=camera, capture=capture, processing=processing,
                              client=client, worker=worker, processor=processor,
                              shared=shared, ready=ready, pyindi=pyindi, image_q=image_q,
                              directory=tmp_path, error_q=error_q, control=control)
    finally:
        client.clearMoonComposite()
        while True:
            try:
                job = image_q.get(timeout=0.05)
            except queue.Empty:
                break
            Path(job['filename']).unlink(missing_ok=True)
            if job.get('moon_composite'):
                Path(job['moon_composite']['filename']).unlink(missing_ok=True)
        for image_queue in queues:
            image_queue.close()
            image_queue.join_thread()
        for filename in tmp_path.glob('*.fit'):
            filename.unlink(missing_ok=True)


def blob(exposure, value=180, headers=None, dtype=numpy.uint16):
    data = numpy.zeros((80, 80), dtype)
    yy, xx = numpy.ogrid[:80, :80]
    data[(xx - 40) ** 2 + (yy - 40) ** 2 <= 12 ** 2] = value * (2 ** (numpy.iinfo(dtype).bits - 8))
    stream = io.BytesIO()
    header = fits.Header({'EXPTIME': exposure}) if exposure is not None else fits.Header()
    if headers:
        header.update(headers)
    fits.PrimaryHDU(data, header=header).writeto(stream)
    return SimpleNamespace(getblobdata=stream.getvalue)


def test_duplicate_short_blob_is_not_cached_twice(runtime, caplog):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    first = client.moon_composite_frame
    client.newBLOB(blob(0.001, value=100))
    assert client.moon_composite_frame is first
    assert Path(first['filename']).exists()
    assert 'duplicate' in caplog.text.lower()


def test_delayed_short_blob_cannot_become_the_long_exposure(runtime, caplog):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    client.moon_composite_short = False
    client.setCcdExposure(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    assert client.moon_composite_waiting_blob
    assert client.moon_composite_frame is not None
    assert 'exposure' in caplog.text.lower()
    client.newBLOB(blob(0.1, value=255))
    job = runtime.image_q.get(timeout=2)
    try:
        assert job['exposure'] == 0.1
        assert job['moon_composite']['exposure'] == 0.001
    finally:
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()


def test_simultaneous_duplicate_callbacks_accept_only_one_file(runtime):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(client.newBLOB, [blob(0.001), blob(0.001)]))
    assert len(list(runtime.directory.glob('*.fit'))) == 1
    assert client.moon_composite_frame is not None


@pytest.mark.parametrize('invalid_timestamp', [False, True])
def test_stale_same_exposure_timestamp_cannot_enter_a_new_bracket(runtime, caplog, invalid_timestamp):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    stale = datetime.fromtimestamp(client.exposureStartTime - 60, tz=timezone.utc).isoformat()
    if invalid_timestamp:
        stale = 'not-a-timestamp'
    client.newBLOB(blob(0.001, headers={'DATE-OBS': stale}))
    assert client.moon_composite_waiting_blob
    assert client.moon_composite_frame is None
    assert not list(runtime.directory.glob('*.fit'))
    assert 'DATE-OBS' in caplog.text


def test_callback_before_exposure_command_returns_has_complete_metadata(runtime, monkeypatch):
    client = runtime.client
    observed = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        def immediate(device, control, values, **kwargs):
            exposure = values['CCD_EXPOSURE_VALUE']
            observed.append(client._expUtils.EXPOSURE_CURRENT)
            executor.submit(client.newBLOB, blob(exposure)).result(timeout=2)
            return runtime.control
        monkeypatch.setattr(client, 'set_number', immediate)
        runtime.worker.shoot(0.1, 75, 1, sync=False)
        assert client.moon_composite_frame['exposure'] == 0.001
        client.moon_composite_short = False
        client.setCcdExposure(0.1, 75, 1, sync=False)
    job = runtime.image_q.get(timeout=2)
    try:
        assert observed == [0.001, 0.1]
        assert job['exposure'] == 0.1
        assert job['moon_composite']['exposure'] == 0.001
    finally:
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()


@pytest.mark.parametrize('whole_seconds', [False, True])
def test_current_utc_timestamp_is_accepted(runtime, whole_seconds):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    captured_at = datetime.fromtimestamp(runtime.client.exposureStartTime, tz=timezone.utc)
    date_obs = captured_at.strftime('%Y-%m-%dT%H:%M:%S') if whole_seconds else captured_at.isoformat()
    runtime.client.newBLOB(blob(0.001, headers={'DATE-OBS': date_obs}))
    assert runtime.client.moon_composite_frame is not None


def test_failed_queue_transfer_keeps_short_owned_until_cleanup(runtime, caplog):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    short_filename = Path(client.moon_composite_frame['filename'])
    client.moon_composite_short = False
    client.setCcdExposure(0.1, 75, 1, sync=False)
    client.image_q = SimpleNamespace(put=mock.Mock(side_effect=ValueError('queue closed')))
    client.newBLOB(blob(0.1))
    assert client.moon_composite_frame is not None
    assert client.moon_composite_waiting_blob
    assert list(runtime.directory.glob('*.fit')) == [short_filename]
    assert 'queue' in caplog.text.lower()
    client.clearMoonComposite()
    assert not short_filename.exists()


@pytest.mark.parametrize('reported', ['not-a-number', 'NaN', -1, 0.1])
def test_invalid_short_exptime_waits_for_correct_frame(runtime, reported, caplog):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    runtime.client.newBLOB(blob(None, headers={'EXPTIME': reported}))
    assert runtime.client.moon_composite_frame is None
    assert runtime.client.moon_composite_waiting_blob
    assert not list(runtime.directory.glob('*.fit'))
    assert 'EXPTIME' in caplog.text
    runtime.client.newBLOB(blob(0.0010005))
    assert runtime.client.moon_composite_frame is not None


def test_malformed_blob_and_failed_write_leave_no_partial_files(runtime, caplog):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    runtime.client.newBLOB(SimpleNamespace(getblobdata=lambda: b'not FITS'))
    payload = blob(0.001)
    with mock.patch.object(fits.HDUList, 'writeto', side_effect=OSError('disk full')):
        runtime.client.newBLOB(payload)
    assert runtime.client.moon_composite_frame is None
    assert runtime.client.moon_composite_waiting_blob
    assert not list(runtime.directory.glob('*.fit'))
    assert 'Invalid camera FITS BLOB' in caplog.text
    runtime.client.newBLOB(payload)
    assert runtime.client.moon_composite_frame is not None


class Clock:
    def __init__(self):
        self.now = self.started = time.time()
        self.events = []
        self.sequence = 0

    def time(self):
        return self.now

    def later(self, delay, callback):
        self.sequence += 1
        heapq.heappush(self.events, (self.now + delay, self.sequence, callback))

    def sleep(self, delay):
        end = self.now + delay
        assert end - self.started < 30, 'Capture loop did not finish within the simulated time budget'
        while self.events and self.events[0][0] <= end:
            self.now, _, callback = heapq.heappop(self.events)
            callback()
        self.now = end


def run_capture(runtime, monkeypatch, *, legacy=False, missing=None, shutdown_after_short=False,
                missing_metadata=False, short_headers=None, short_value=180, short_dtype=numpy.uint16):
    clock = Clock()
    worker, client = runtime.worker, runtime.client
    monkeypatch.setattr(runtime.capture, 'time', clock)
    monkeypatch.setattr(runtime.camera, 'time', clock)
    monkeypatch.setattr(runtime.capture, 'signal', SimpleNamespace(
        SIGHUP=1, SIGTERM=2, SIGINT=3, SIGALRM=4, signal=lambda *args: None,
    ))
    for method in (
        'detectNight', '_initialize', '_pre_run_tasks', 'getCcdTemperature',
        'getTelescopeRaDec', 'getGpsPosition', 'capture_pre_hook',
    ):
        monkeypatch.setattr(worker, method, lambda: None)
    monkeypatch.setattr(worker, 'reconfigureCcd', lambda: setattr(worker, 'reconfigure_camera', False))
    worker._dateCalcs = SimpleNamespace(
        getNextDayNightTransition=lambda: datetime.fromtimestamp(clock.time() + 86400),
    )
    worker.exposure_timeout = 0.2 if missing else 10
    worker.periodic_tasks_time = clock.time() + 300
    requests, deliveries = [], []

    def start(device, control, values, **kwargs):
        exposure = values['CCD_EXPOSURE_VALUE']
        phase = 'short' if exposure == 0.001 else 'long'
        requests.append((phase, clock.time()))
        runtime.ready.value = False
        clock.later(exposure, lambda: setattr(runtime.ready, 'value', True))
        if missing != phase:
            def deliver():
                omit_exposure = missing_metadata is True or missing_metadata == phase
                payload = blob(None if omit_exposure else exposure,
                               value=short_value if phase == 'short' else 255,
                               headers=short_headers if phase == 'short' else None,
                               dtype=short_dtype if phase == 'short' else numpy.uint16)
                if legacy:
                    client.newBLOB(payload)
                else:
                    class Property(list):
                        def getType(self):
                            return runtime.pyindi.INDI_BLOB

                        def getDeviceName(self):
                            return 'test-camera'
                    client.updateProperty(Property([payload]))
                deliveries.append((phase, clock.time()))
                if phase == 'long' or shutdown_after_short:
                    worker._shutdown = True
            clock.later(exposure + 0.25, deliver)
        return runtime.control

    monkeypatch.setattr(client, 'set_number', start)
    worker.run()
    with pytest.raises(queue.Empty):
        runtime.error_q.get(timeout=0.05)
    return requests, deliveries


def ingest_long(runtime, job):
    camera = SimpleNamespace(id=7, name='test-camera', uuid='test', owner='', location='',
                             lensFocalLength=2.5, lensFocalRatio=2.0, data={})
    processor = runtime.processor
    i_ref = processor._add(
        job['filename'], job['exposure'], job['gain'], job['binning'],
        datetime.fromtimestamp(job['exp_time']), job['exp_elapsed'], camera,
    )
    processor.image = numpy.right_shift(processor._debayer(i_ref), 8).astype(numpy.uint8)
    return i_ref


@pytest.mark.parametrize('legacy', [False, True])
def test_capture_loop_waits_for_blobs_then_processes_one_pair(runtime, monkeypatch, legacy):
    requests, deliveries = run_capture(runtime, monkeypatch, legacy=legacy)
    assert [phase for phase, _ in requests] == ['short', 'long']
    assert requests[1][1] >= deliveries[0][1]
    job = runtime.image_q.get(timeout=2)
    try:
        assert job['moon_composite_id'] == job['moon_composite']['moon_composite_id'] == 1
        processor = runtime.processor
        i_ref = ingest_long(runtime, job)
        original = i_ref.hdulist[0].data.copy()
        try:
            processor.composite_moon(job['moon_composite'], i_ref)
            assert processor.image[40, 40] == 180
            numpy.testing.assert_array_equal(original, i_ref.hdulist[0].data)
        finally:
            i_ref.hdulist.close()
        with pytest.raises(queue.Empty):
            runtime.image_q.get(timeout=0.05)
    finally:
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()


@pytest.mark.parametrize('missing', ['short', 'long'])
def test_timed_out_bracket_disconnects_and_removes_incomplete_files(runtime, monkeypatch, missing):
    run_capture(runtime, monkeypatch, missing=missing)
    runtime.client.abortCcdExposure.assert_called_once()
    assert runtime.client.disconnected
    assert runtime.client.moon_composite_frame is None
    assert not list(runtime.directory.glob('*.fit'))
    runtime.client.newBLOB(blob(0.001))
    assert not list(runtime.directory.glob('*.fit'))
    with pytest.raises(queue.Empty):
        runtime.image_q.get(timeout=0.05)


def test_orderly_shutdown_after_short_removes_short_without_starting_long(runtime, monkeypatch):
    requests, _ = run_capture(runtime, monkeypatch, shutdown_after_short=True)
    assert [phase for phase, _ in requests] == ['short']
    assert not list(runtime.directory.glob('*.fit'))
    assert runtime.client.disconnected


@pytest.mark.parametrize('phase', [True, 'short', 'long'])
def test_missing_exptime_keeps_original_sky_and_reports_driver_limitation(runtime, monkeypatch, caplog, phase):
    run_capture(runtime, monkeypatch, missing_metadata=phase)
    job = runtime.image_q.get(timeout=2)
    try:
        assert 'moon_composite' not in job
        assert 'driver did not supply EXPTIME' in caplog.text
        assert len(list(runtime.directory.glob('*.fit'))) == 1
    finally:
        Path(job['filename']).unlink()


@pytest.mark.parametrize('night', [False, True])
def test_capture_loop_outside_moon_mode_queues_only_a_normal_frame(runtime, monkeypatch, night):
    runtime.worker.night = night
    runtime.worker.moonmode = False
    runtime.shared.night[constants.NIGHT_NIGHT] = int(night)
    runtime.shared.night[constants.NIGHT_MOONMODE] = 0
    runtime.worker.config['DAYTIME_CAPTURE'] = True
    requests, _ = run_capture(runtime, monkeypatch)
    assert [phase for phase, _ in requests] == ['long']
    job = runtime.image_q.get(timeout=2)
    try:
        assert 'moon_composite' not in job
        assert 'moon_composite_id' not in job
        assert len(list(runtime.directory.glob('*.fit'))) == 1
    finally:
        Path(job['filename']).unlink()


@pytest.mark.parametrize('header,value,dtype', [
    ('XBAYROFF', 1, numpy.uint16), ('YBAYROFF', 1, numpy.uint16),
    ('XORGSUBF', 2, numpy.uint16), ('YORGSUBF', 2, numpy.uint16),
    ('ROWORDER', 'BOTTOM-UP', numpy.uint16), ('BAYERPAT', 'RGGB', numpy.uint16),
    ('BITPIX', 8, numpy.uint8),
])
def test_geometry_mismatch_keeps_sky_and_names_the_problem(runtime, monkeypatch, caplog, header, value, dtype):
    run_capture(runtime, monkeypatch, short_headers={header: value}, short_dtype=dtype)
    job = runtime.image_q.get(timeout=2)
    i_ref = ingest_long(runtime, job)
    before = runtime.processor.image.copy()
    try:
        runtime.processor.composite_moon(job['moon_composite'], i_ref)
        numpy.testing.assert_array_equal(before, runtime.processor.image)
        assert 'Moon composite skipped' in caplog.text
        assert header in caplog.text or (header == 'BAYERPAT' and 'Bayer patterns' in caplog.text)
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()


def test_clipped_short_frame_keeps_original_and_actionable_log(runtime, monkeypatch, caplog):
    run_capture(runtime, monkeypatch, short_value=255)
    job = runtime.image_q.get(timeout=2)
    i_ref = ingest_long(runtime, job)
    before = runtime.processor.image.copy()
    try:
        runtime.processor.composite_moon(job['moon_composite'], i_ref)
        numpy.testing.assert_array_equal(before, runtime.processor.image)
        assert 'short exposure is clipped; reduce EXPOSURE' in caplog.text
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()
