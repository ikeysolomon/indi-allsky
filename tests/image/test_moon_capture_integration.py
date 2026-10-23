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
from threading import Event
import time
from types import ModuleType, SimpleNamespace
from unittest import mock

from astropy.io import fits
from flask import Flask
import numpy
import pytest

from indi_allsky import constants
from indi_allsky import stretch as stretch_classes


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
    clients = []

    def create_client():
        client = camera.IndiClient(config, image_q, shared.position, shared.exposure,
                                  shared.gain, shared.binning, shared.night)
        clients.append(client)
        client.gain, client.binning, client.camera_id = 75, 1, 7
        client.ccd_device = SimpleNamespace(getDeviceName=lambda: 'test-camera')
        client._indi_allsky_camera_name = 'test-camera'
        client.disableCcdCooler = mock.Mock()
        client.abortCcdExposure = mock.Mock()
        ready = SimpleNamespace(value=True)
        control = SimpleNamespace(
            getState=lambda: pyindi.IPS_OK if ready.value else pyindi.IPS_BUSY,
            getName=lambda: 'CCD_EXPOSURE',
        )
        client.ccd_device.getNumber = lambda name: control
        client.set_number = mock.Mock(return_value=control)
        return client, ready, control

    def create_runtime():
        client, ready, control = create_client()
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
        return SimpleNamespace(camera=camera, capture=capture, processing=processing,
                               client=client, worker=worker, processor=processor,
                               shared=shared, ready=ready, pyindi=pyindi, image_q=image_q,
                               directory=tmp_path, error_q=error_q, control=control,
                               restart=create_runtime, new_client=create_client)

    processor = processing.ImageProcessor(
        config, shared.position, shared.exposure, shared.gain, shared.binning,
        shared.sensors_temp, shared.sensors_user, shared.night, shared.astro,
    )
    try:
        yield create_runtime()
    finally:
        for client in clients:
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
    first = client._moon_capture.short_frame
    client.newBLOB(blob(0.001, value=100))
    assert client._moon_capture.short_frame is first
    assert Path(first['filename']).exists()
    assert 'duplicate' in caplog.text.lower()


def test_one_operation_owns_short_long_request_and_completion(runtime):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    operation = client._moon_capture
    assert operation.id == 1
    assert operation.long_request == (0.1, 75, 1)
    assert (operation.phase, operation.status) == ('short', 'waiting')
    client.newBLOB(blob(0.001))
    short = operation.short_frame
    assert operation.status == 'received'
    assert client.continueMoonComposite()
    assert client._moon_capture is operation
    assert (operation.phase, operation.status) == ('long', 'waiting')
    client.newBLOB(blob(0.1))
    job = runtime.image_q.get(timeout=2)
    try:
        assert job['moon_composite'] == short
        assert job['moon_composite_id'] == job['moon_composite']['moon_composite_id'] == operation.id
        assert operation.status == 'complete'
        assert operation.short_frame is None
        client.clearMoonComposite()
        assert operation.status == 'complete'
        client.beginMoonComposite(0.2, 75, 1)
        assert client._moon_capture.id == 2
        assert client._moon_capture is not operation
        assert operation.status == 'complete'
    finally:
        Path(job['filename']).unlink()
        Path(short['filename']).unlink()


def test_delayed_short_blob_cannot_become_the_long_exposure(runtime, caplog):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    assert client.continueMoonComposite()
    client.newBLOB(blob(0.001))
    assert client._moon_capture.status == 'waiting'
    assert client._moon_capture.short_frame is not None
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
    assert client._moon_capture.short_frame is not None


@pytest.mark.parametrize('phase', ['prepared', 'short-waiting', 'short-received', 'long-waiting', 'normal'])
def test_new_capture_cannot_replace_an_unfinished_operation(runtime, phase):
    client = runtime.client
    if phase == 'prepared':
        client.beginMoonComposite(0.1, 75, 1)
    elif phase == 'normal':
        client.setCcdExposure(0.0005, 75, 1, sync=False)
    else:
        runtime.worker.shoot(0.1, 75, 1, sync=False)
        if phase != 'short-waiting':
            client.newBLOB(blob(0.001))
        if phase == 'long-waiting':
            assert client.continueMoonComposite()
    operation = client._moon_capture
    with pytest.raises(runtime.camera.CameraException, match='in progress'):
        client.beginMoonComposite(0.2, 100, 2)
    if phase != 'prepared':
        with pytest.raises(runtime.camera.CameraException, match='Complete or clear'):
            client.setCcdExposure(0.2, 100, 2, sync=False)
    assert client._moon_capture is operation
    assert client.gain == 75
    assert client.binning == 1


@pytest.mark.parametrize('operation', ['clear', 'long', 'status', 'pending'])
def test_bracket_transitions_and_snapshots_wait_for_callback_lock(runtime, operation):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    short_file = Path(client._moon_capture.short_frame['filename'])
    started = Event()

    def transition():
        started.set()
        if operation == 'clear':
            return client.clearMoonComposite()
        if operation == 'long':
            return client.continueMoonComposite()
        if operation == 'status':
            return client.getCcdExposureStatus()
        return client.moonCompositeInProgress()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with client._moon_composite_lock:
            pending = executor.submit(transition)
            assert started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.05)
        result = pending.result(timeout=2)
    if operation == 'clear':
        assert client._moon_capture.status == 'cancelled'
        assert client._moon_capture.short_frame is None
        assert not short_file.exists()
    elif operation == 'long':
        assert client._moon_capture.phase == 'long'
        assert client._moon_capture.short_frame is not None
        assert short_file.exists()
    elif operation == 'pending':
        assert result is True
    else:
        assert result == (True, 'OK')


@pytest.mark.parametrize('invalid_timestamp', [False, True])
def test_stale_same_exposure_timestamp_cannot_enter_a_new_bracket(runtime, caplog, invalid_timestamp):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    stale = datetime.fromtimestamp(client.exposureStartTime - 60, tz=timezone.utc).isoformat()
    if invalid_timestamp:
        stale = 'not-a-timestamp'
    client.newBLOB(blob(0.001, headers={'DATE-OBS': stale}))
    assert client._moon_capture.status == 'waiting'
    assert client._moon_capture.short_frame is None
    assert not list(runtime.directory.glob('*.fit'))
    assert 'DATE-OBS' in caplog.text


@pytest.mark.parametrize('has_timestamp', [False, True])
def test_replayed_same_exposure_frame_is_indistinguishable_without_date_obs(
    runtime, monkeypatch, caplog, has_timestamp,
):
    client = runtime.client
    clock = SimpleNamespace(now=time.time())
    monkeypatch.setattr(runtime.camera, 'time', SimpleNamespace(time=lambda: clock.now))
    old_headers = {'DATE-OBS': datetime.fromtimestamp(clock.now, tz=timezone.utc).isoformat()} if has_timestamp else None
    old_blob = blob(0.001, value=100, headers=old_headers)
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    first_operation = client._moon_capture
    client.newBLOB(old_blob)
    first_filename = Path(first_operation.short_frame['filename'])
    client.clearMoonComposite()
    assert not first_filename.exists()

    clock.now += 60
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    second_operation = client._moon_capture
    assert second_operation.id == first_operation.id + 1
    client.newBLOB(old_blob)
    if has_timestamp:
        assert second_operation.status == 'waiting'
        assert second_operation.short_frame is None
        assert 'Ignoring stale Moon bracket' in caplog.text
    else:
        assert second_operation.status == 'received'
        assert second_operation.short_frame['moon_composite_id'] == second_operation.id
        assert 'Ignoring stale Moon bracket' not in caplog.text

    current_headers = {'DATE-OBS': datetime.fromtimestamp(clock.now, tz=timezone.utc).isoformat()} if has_timestamp else None
    client.newBLOB(blob(0.001, value=180, headers=current_headers))
    assert client.continueMoonComposite()
    client.newBLOB(blob(0.1, headers=current_headers))
    job = runtime.image_q.get(timeout=2)
    try:
        assert job['moon_composite_id'] == job['moon_composite']['moon_composite_id'] == second_operation.id
        with fits.open(job['moon_composite']['filename']) as hdulist:
            assert ('DATE-OBS' in hdulist[0].header) is has_timestamp
            assert hdulist[0].header['EXPTIME'] == 0.001
            assert hdulist[0].data[40, 40] == (180 if has_timestamp else 100) * 256
    finally:
        Path(job['filename']).unlink()
        Path(job['moon_composite']['filename']).unlink()


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
        assert client._moon_capture.short_frame['exposure'] == 0.001
        assert client.continueMoonComposite()
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
    assert runtime.client._moon_capture.short_frame is not None


def test_failed_queue_transfer_keeps_short_owned_until_cleanup(runtime, caplog):
    client = runtime.client
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    client.newBLOB(blob(0.001))
    short_filename = Path(client._moon_capture.short_frame['filename'])
    assert client.continueMoonComposite()
    client.image_q = SimpleNamespace(put=mock.Mock(side_effect=ValueError('queue closed')))
    client.newBLOB(blob(0.1))
    assert client._moon_capture.short_frame is not None
    assert client._moon_capture.status == 'waiting'
    assert list(runtime.directory.glob('*.fit')) == [short_filename]
    assert 'queue' in caplog.text.lower()
    client.clearMoonComposite()
    assert not short_filename.exists()


@pytest.mark.parametrize('reported', ['not-a-number', 'NaN', -1, 0.1])
def test_invalid_short_exptime_waits_for_correct_frame(runtime, reported, caplog):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    runtime.client.newBLOB(blob(None, headers={'EXPTIME': reported}))
    assert runtime.client._moon_capture.short_frame is None
    assert runtime.client._moon_capture.status == 'waiting'
    assert not list(runtime.directory.glob('*.fit'))
    assert 'EXPTIME' in caplog.text
    runtime.client.newBLOB(blob(0.0010005))
    assert runtime.client._moon_capture.short_frame is not None


def test_malformed_blob_and_failed_write_leave_no_partial_files(runtime, caplog):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    runtime.client.newBLOB(SimpleNamespace(getblobdata=lambda: b'not FITS'))
    payload = blob(0.001)
    with mock.patch.object(fits.HDUList, 'writeto', side_effect=OSError('disk full')):
        runtime.client.newBLOB(payload)
    assert runtime.client._moon_capture.short_frame is None
    assert runtime.client._moon_capture.status == 'waiting'
    assert not list(runtime.directory.glob('*.fit'))
    assert 'Invalid camera FITS BLOB' in caplog.text
    runtime.client.newBLOB(payload)
    assert runtime.client._moon_capture.short_frame is not None


@pytest.mark.parametrize('damage', ['card-order', 'truncated-data'])
def test_structurally_malformed_fits_waits_for_valid_replacement(runtime, caplog, damage):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    payload = bytearray(blob(0.001).getblobdata())
    if damage == 'card-order':
        payload[80:160], payload[160:240] = payload[160:240], payload[80:160]
        with fits.open(io.BytesIO(payload)) as hdulist:
            with pytest.raises(fits.verify.VerifyError):
                hdulist.verify('exception')
    else:
        payload = payload[:2888]
    runtime.client.newBLOB(SimpleNamespace(getblobdata=lambda: bytes(payload)))
    assert runtime.client._moon_capture.short_frame is None
    assert runtime.client._moon_capture.status == 'waiting'
    assert not list(runtime.directory.glob('*.fit'))
    assert 'Invalid camera FITS BLOB' in caplog.text
    runtime.client.newBLOB(blob(0.001))
    assert runtime.client._moon_capture.short_frame is not None


def test_unexpected_fits_programming_error_is_not_silenced(runtime):
    runtime.worker.shoot(0.1, 75, 1, sync=False)
    payload = blob(0.001)
    with mock.patch.object(fits.HDUList, 'writeto', side_effect=TypeError('unexpected write bug')):
        with pytest.raises(TypeError, match='unexpected write bug'):
            runtime.client.newBLOB(payload)
    assert not list(runtime.directory.glob('*.fit'))


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
                missing_metadata=False, short_headers=None, short_value=180, short_dtype=numpy.uint16,
                reported_exposures=None, normal_frames=1, recovery_busy=0):
    clock = Clock()
    worker, original_client = runtime.worker, runtime.client
    monkeypatch.setattr(runtime.capture, 'time', clock)
    monkeypatch.setattr(runtime.camera, 'time', clock)
    monkeypatch.setattr(runtime.capture, 'signal', SimpleNamespace(
        SIGHUP=1, SIGTERM=2, SIGINT=3, SIGALRM=4, signal=lambda *args: None,
    ))
    for method in (
        'detectNight', '_pre_run_tasks', 'getCcdTemperature',
        'getTelescopeRaDec', 'getGpsPosition', 'capture_pre_hook',
    ):
        monkeypatch.setattr(worker, method, lambda: None)
    monkeypatch.setattr(worker, 'reconfigureCcd', lambda: setattr(worker, 'reconfigure_camera', False))
    worker._dateCalcs = SimpleNamespace(
        getNextDayNightTransition=lambda: datetime.fromtimestamp(clock.time() + 86400),
    )
    worker.exposure_timeout = 0.2 if missing or reported_exposures else 10
    worker.periodic_tasks_time = clock.time() + 300
    requests, deliveries = [], []
    completed = 0
    initialized = False

    def start(client, ready, exposure_control, device, control, values, **kwargs):
        exposure = values['CCD_EXPOSURE_VALUE']
        phase = 'short' if exposure == 0.001 else 'long'
        requests.append((phase, clock.time()))
        ready.value = False
        clock.later(exposure, lambda: setattr(ready, 'value', True))
        if missing != phase or client is not original_client:
            def deliver():
                nonlocal completed
                omit_exposure = missing_metadata is True or missing_metadata == phase
                reported_exposure = (reported_exposures or {}).get(phase, exposure)
                payload = blob(None if omit_exposure else reported_exposure,
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
                capture = client._moon_capture
                if phase == 'long' and (capture is None or capture.status == 'complete'):
                    completed += 1
                if completed >= normal_frames or shutdown_after_short:
                    worker._shutdown = True
            clock.later(exposure + 0.25, deliver)
        return exposure_control

    def install_transport():
        monkeypatch.setattr(runtime.client, 'set_number', partial(
            start, runtime.client, runtime.ready, runtime.control,
        ))

    def initialize():
        nonlocal initialized
        if initialized:
            runtime.client, runtime.ready, runtime.control = runtime.new_client()
            worker.indiclient = runtime.client
            if recovery_busy:
                runtime.ready.value = False
                runtime.recovery_ready_at = clock.time() + recovery_busy
                clock.later(recovery_busy, lambda: setattr(runtime.ready, 'value', True))
            install_transport()
        initialized = True

    monkeypatch.setattr(worker, '_initialize', initialize)
    install_transport()
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
    processor.image_list = [i_ref]
    processor.image = numpy.right_shift(processor._debayer(i_ref), 8).astype(numpy.uint8)
    return i_ref


@pytest.mark.parametrize('mode', stretch_classes.__all__)
@pytest.mark.parametrize('bits', [8, 12, 16])
@pytest.mark.parametrize('color', [False, True])
def test_moon_roi_reuses_current_sky_transfer_before_8bit_conversion(runtime, monkeypatch, mode, bits, color):
    run_capture(runtime, monkeypatch, short_dtype=numpy.uint8 if bits == 8 else numpy.uint16)
    job = runtime.image_q.get(timeout=2)
    short_filename = Path(job['moon_composite']['filename'])
    dtype = numpy.uint8 if bits == 8 else numpy.uint16
    scale = 2 ** (bits - 8)
    short_data = numpy.zeros((80, 80), dtype)
    yy, xx = numpy.ogrid[:80, :80]
    disk = (xx - 40) ** 2 + (yy - 40) ** 2 <= 12 ** 2
    short_data[disk] = 180 * scale
    long_data = numpy.broadcast_to((15 + numpy.arange(80) // 4) * scale, (80, 80)).astype(dtype).copy()
    long_data[disk] = 255 * scale
    if color:
        short_data = numpy.stack([short_data] * 3)
        long_data = numpy.stack([long_data] * 3)
    fits.PrimaryHDU(short_data, header=fits.Header({'EXPTIME': 0.001})).writeto(short_filename, overwrite=True)
    fits.PrimaryHDU(long_data, header=fits.Header({'EXPTIME': 0.1})).writeto(job['filename'], overwrite=True)
    short_bytes = short_filename.read_bytes()
    with fits.conf.set_temp('use_memmap', False):
        i_ref = ingest_long(runtime, job)
    processor = runtime.processor
    processor.max_bit_depth = bits
    processor.config['IMAGE_STRETCH'] = {'MOONMODE': True}
    operator = getattr(stretch_classes, mode)(processor.config, mask={1: None})
    processor._stretch_o = operator
    processor.image = processor._debayer(i_ref)
    original_fits = i_ref.hdulist[0].data.copy()
    try:
        processor.stretch(moon_composite=True)
        expected = operator.apply_last_stretch(numpy.moveaxis(short_data, 0, -1) if color else short_data)
        if bits != 8:
            expected = numpy.right_shift(expected, bits - 8).astype(numpy.uint8)
        processor.convert_16bit_to_8bit()
        before = processor.image.copy()
        monkeypatch.setattr(operator, 'stretch', mock.Mock(side_effect=AssertionError('ROI must not refit stretch')))
        processor.composite_moon(job['moon_composite'], i_ref)
        numpy.testing.assert_array_equal(processor.image[40, 40], expected[40, 40])
        assert numpy.any(expected[40, 40] != 180)
        numpy.testing.assert_array_equal(processor.image[:20], before[:20])
        numpy.testing.assert_array_equal(i_ref.hdulist[0].data, original_fits)
        assert short_filename.read_bytes() == short_bytes
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink(missing_ok=True)
        short_filename.unlink()


@pytest.mark.parametrize('flip', [False, True])
def test_split_display_preserves_linear_side_and_divider(runtime, monkeypatch, flip):
    run_capture(runtime, monkeypatch)
    job = runtime.image_q.get(timeout=2)
    i_ref = ingest_long(runtime, job)
    processor = runtime.processor
    processor.config['IMAGE_STRETCH'] = {'MOONMODE': True, 'SPLIT': True}
    processor.config['IMAGE_FLIP_H'] = flip
    operator = stretch_classes.mode2_mtf(processor.config)
    processor._stretch_o = operator
    processor.image = processor._debayer(i_ref)
    try:
        processor.stretch(moon_composite=True)
        expected = int(operator.apply_last_stretch(numpy.array([180 * 256], numpy.uint16))[0]) >> 8
        processor.convert_16bit_to_8bit()
        processor.composite_moon(job['moon_composite'], i_ref)
        assert processor.image[40, 40] == 0
        assert processor.image[40, 39] == (expected if flip else 180)
        assert processor.image[40, 41] == (180 if flip else expected)
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink(missing_ok=True)
        Path(job['moon_composite']['filename']).unlink()


@pytest.mark.parametrize('gate', ['moonmode', 'focus', 'no-operator', 'disabled', 'not-moon', 'day', 'no-pair', 'clahe'])
def test_skipped_stretch_cannot_reuse_previous_frames_transfer(runtime, gate):
    processor = runtime.processor
    processor.config['IMAGE_STRETCH'] = {'MOONMODE': True}
    processor._stretch_o = stretch_classes.mode2_mtf(processor.config)
    processor.image = numpy.full((80, 80), 100, numpy.uint8)
    processor.max_bit_depth = 8
    processor.getLatestImage = lambda: SimpleNamespace(binning=1, image_bitpix=8)
    processor.stretch(moon_composite=True)
    assert processor._moon_display_lut is not None
    previous = processor._stretch_o
    if gate == 'moonmode':
        processor.config['IMAGE_STRETCH']['MOONMODE'] = False
    elif gate == 'focus':
        processor.focus_mode = True
    elif gate == 'disabled':
        processor.config['MOON_COMPOSITE']['ENABLE'] = False
    elif gate == 'not-moon':
        runtime.shared.night[constants.NIGHT_MOONMODE] = 0
    elif gate == 'day':
        runtime.shared.night[constants.NIGHT_NIGHT] = 0
    elif gate == 'clahe':
        processor.config.update(CONTRAST_ENHANCE_16BIT=True, NIGHT_CONTRAST_ENHANCE=True)
    elif gate == 'no-operator':
        processor._stretch_o = None
    processor.stretch(moon_composite=gate != 'no-pair')
    assert processor._moon_display_lut is None
    if gate != 'no-operator':
        assert not previous._display_luts
        assert not previous.record_display_transfer


def test_invalid_roi_transfer_range_keeps_sky_and_reports_bit_depth(runtime, monkeypatch, caplog):
    run_capture(runtime, monkeypatch)
    job = runtime.image_q.get(timeout=2)
    i_ref = ingest_long(runtime, job)
    processor = runtime.processor
    processor._moon_display_lut = numpy.arange(256, dtype=numpy.uint8)
    before = processor.image.copy()
    try:
        processor.composite_moon(job['moon_composite'], i_ref)
        numpy.testing.assert_array_equal(processor.image, before)
        assert 'short ROI exceeds sky stretch range; check sensor bit depth' in caplog.text
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink(missing_ok=True)
        Path(job['moon_composite']['filename']).unlink()


@pytest.mark.parametrize('sixteen_bit,night_contrast', [(True, True), (True, False), (False, True)])
def test_contrast_compatibility_only_skips_active_16bit_night_brackets(
    runtime, monkeypatch, caplog, sixteen_bit, night_contrast,
):
    runtime.worker.config.update(
        CONTRAST_ENHANCE_16BIT=sixteen_bit, NIGHT_CONTRAST_ENHANCE=night_contrast,
    )
    requests, _ = run_capture(runtime, monkeypatch)
    job = runtime.image_q.get(timeout=2)
    blocked = sixteen_bit and night_contrast
    try:
        assert [phase for phase, _ in requests] == (['long'] if blocked else ['short', 'long'])
        assert ('moon_composite' in job) is not blocked
        assert runtime.worker.config['MOON_COMPOSITE']['ENABLE']
        assert not runtime.worker.moon_composite_suspended
        if blocked:
            assert '16-bit night contrast enhancement (CLAHE) is active' in caplog.text
            assert len(list(runtime.directory.glob('*.fit'))) == 1
    finally:
        Path(job['filename']).unlink()
        if job.get('moon_composite'):
            Path(job['moon_composite']['filename']).unlink()


@pytest.mark.parametrize('color', [False, True])
def test_queued_pair_keeps_clahe_sky_without_reading_short_roi(runtime, monkeypatch, caplog, color):
    run_capture(runtime, monkeypatch)
    job = runtime.image_q.get(timeout=2)
    i_ref = ingest_long(runtime, job)
    processor = runtime.processor
    processor.config.update(CONTRAST_ENHANCE_16BIT=True, NIGHT_CONTRAST_ENHANCE=True)
    processor.image = processor._debayer(i_ref)
    if color:
        processor.image = numpy.repeat(processor.image[:, :, None], 3, axis=2)
    original_fits = i_ref.hdulist[0].data.copy()
    try:
        processor.contrast_clahe_16bit()
        processor.convert_16bit_to_8bit()
        before = processor.image.copy()
        with mock.patch.object(fits, 'open', side_effect=AssertionError('incompatible composite must not read ROI')):
            processor.composite_moon(job['moon_composite'], i_ref)
        numpy.testing.assert_array_equal(processor.image, before)
        numpy.testing.assert_array_equal(i_ref.hdulist[0].data, original_fits)
        assert '16-bit night contrast enhancement (CLAHE) is active' in caplog.text
    finally:
        i_ref.hdulist.close()
        Path(job['filename']).unlink(missing_ok=True)
        Path(job['moon_composite']['filename']).unlink()


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
def test_timed_out_bracket_resets_connection_and_resumes_normal_capture(runtime, monkeypatch, missing):
    original_client = runtime.client
    requests, _ = run_capture(runtime, monkeypatch, missing=missing)
    original_client.abortCcdExposure.assert_called_once()
    assert original_client.disconnected
    assert runtime.client is not original_client
    assert original_client._moon_capture.short_frame is None
    assert runtime.worker.moon_composite_suspended
    assert [phase for phase, _ in requests] == (
        ['short', 'long'] if missing == 'short' else ['short', 'long', 'long']
    )
    job = runtime.image_q.get(timeout=2)
    try:
        assert 'moon_composite' not in job
        assert job['exposure'] == 0.1
        assert len(list(runtime.directory.glob('*.fit'))) == 1
        original_client.newBLOB(blob(0.001))
        with pytest.raises(queue.Empty):
            runtime.image_q.get(timeout=0.05)
        assert len(list(runtime.directory.glob('*.fit'))) == 1
    finally:
        Path(job['filename']).unlink()


def test_recovery_waits_for_idle_camera_before_requesting_ordinary_exposure(runtime, monkeypatch):
    requests, _ = run_capture(runtime, monkeypatch, missing='short', recovery_busy=0.15)
    assert requests[-1][1] >= runtime.recovery_ready_at
    job = runtime.image_q.get(timeout=2)
    try:
        assert 'moon_composite' not in job
        assert job['exposure'] == 0.1
    finally:
        Path(job['filename']).unlink()


def test_recovery_cannot_claim_success_if_camera_stays_busy(runtime, monkeypatch):
    with pytest.raises(runtime.camera.TimeOutException, match='CCD_EXPOSURE'):
        run_capture(runtime, monkeypatch, missing='short', recovery_busy=2)
    message, traceback = runtime.error_q.get(timeout=2)
    assert 'CCD_EXPOSURE' in message
    assert 'TimeOutException' in traceback
    assert runtime.worker.moon_composite_suspended
    assert not list(runtime.directory.glob('*.fit'))
    with pytest.raises(queue.Empty):
        runtime.image_q.get(timeout=0.05)


@pytest.mark.parametrize('missing', ['short', 'long'])
def test_fallback_restores_long_request_after_camera_reinitialization(runtime, monkeypatch, missing):
    create_client = runtime.new_client

    def initialize_with_different_defaults():
        runtime.worker._expUtils.EXPOSURE_NEXT = 0.2
        runtime.worker._expUtils.GAIN_NEXT = 25
        runtime.worker._expUtils.BINNING_NEXT = 2
        return create_client()

    monkeypatch.setattr(runtime, 'new_client', initialize_with_different_defaults)
    run_capture(runtime, monkeypatch, missing=missing)
    job = runtime.image_q.get(timeout=2)
    try:
        assert (job['exposure'], job['gain'], job['binning']) == (0.1, 75, 1)
        assert 'moon_composite' not in job
    finally:
        Path(job['filename']).unlink()


@pytest.mark.parametrize('phase', ['short', 'long'])
@pytest.mark.parametrize('recovery', ['corrected-metadata', 'disabled'])
def test_driver_exptime_mismatch_suspends_until_reload_and_keeps_sky_capture(runtime, monkeypatch, caplog, phase, recovery):
    reported = {phase: 0.0011 if phase == 'short' else 0.11}
    original_client = runtime.client
    requests, _ = run_capture(runtime, monkeypatch, reported_exposures=reported, normal_frames=2)
    assert [name for name, _ in requests] == (
        ['short', 'long', 'long'] if phase == 'short' else ['short', 'long', 'long', 'long']
    )
    original_client.abortCcdExposure.assert_called_once()
    assert original_client.disconnected
    assert original_client._moon_capture.short_frame is None
    assert not original_client.moonCompositeInProgress()
    assert runtime.worker.moon_composite_suspended
    assert runtime.worker.config['MOON_COMPOSITE']['ENABLE'] is True
    for _ in range(2):
        job = runtime.image_q.get(timeout=2)
        assert 'moon_composite' not in job
        assert job['exposure'] == 0.1
        Path(job['filename']).unlink()
    original_client.newBLOB(blob(reported[phase]))
    assert not list(runtime.directory.glob('*.fit'))
    with pytest.raises(queue.Empty):
        runtime.image_q.get(timeout=0.05)
    assert 'EXPTIME' in caplog.text
    assert 'suspended until configuration reload/restart' in caplog.text

    if recovery == 'disabled':
        runtime.worker.config['MOON_COMPOSITE']['ENABLE'] = False
    current = runtime.restart()
    requests, _ = run_capture(
        current, monkeypatch,
        reported_exposures={'long': 0.11} if recovery == 'disabled' else None,
    )
    job = current.image_q.get(timeout=2)
    try:
        current.client.abortCcdExposure.assert_not_called()
        if recovery == 'disabled':
            assert [name for name, _ in requests] == ['long']
            assert 'moon_composite' not in job
            assert 'moon_composite_id' not in job
        else:
            assert [name for name, _ in requests] == ['short', 'long']
            assert job['exposure'] == 0.1
            assert job['moon_composite']['exposure'] == 0.001
        with pytest.raises(queue.Empty):
            current.image_q.get(timeout=0.05)
    finally:
        Path(job['filename']).unlink()
        if job.get('moon_composite'):
            Path(job['moon_composite']['filename']).unlink()


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
