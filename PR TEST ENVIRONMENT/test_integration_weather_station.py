import sys
import types
import time
import threading

from indi_allsky import weather_station as ws
from indi_allsky import Meteorologist


def _make_fake_rdm():
    m = types.SimpleNamespace()

    def compute_weighted_rain_score_with_recent_priority(*args, **kwargs):
        # Always return a high score so push is triggered in tests.
        return {"score": 1.0}

    def linear_slope(series):
        return 0.0

    m.compute_weighted_rain_score_with_recent_priority = compute_weighted_rain_score_with_recent_priority
    m.linear_slope = linear_slope
    return m


def test_enqueue_worker_dispatch():
    # Inject a lightweight fake analysis module to avoid importing heavy
    # dependencies in the test process. Meteorologist will lazily import
    # this module when performing its analysis.
    sys.modules['indi_allsky.rain_detection_models'] = _make_fake_rdm()

    shutdown = threading.Event()

    # Start the meteorologist worker in a background thread using a config
    # that enables analysis so the worker will attach `push_triggered`.
    mqtt_cfg = {'MQTTPUBLISH': {'PUSH_USE_ANALYSIS': True, 'PUSH_SCORE_THRESHOLD': 0.0}}
    t = threading.Thread(target=Meteorologist.run_worker, args=(mqtt_cfg, shutdown), daemon=True)
    t.start()

    # Enqueue a metric that will be processed by the worker.
    metric = {'data': {'rain': 1.0, 'humidity': 50}, '_ts': time.time()}
    assert ws.enqueue_metric(metric)

    try:
        item = ws.get_dispatch(block=True, timeout=5)
    except Exception as e:
        shutdown.set()
        t.join(timeout=2)
        raise AssertionError('No dispatch received') from e

    # Validate dispatched item contains analysis decision
    assert isinstance(item, dict)
    assert 'result' in item
    assert item['result'].get('push_triggered') is True

    # Stop the worker
    shutdown.set()
    t.join(timeout=2)


if __name__ == '__main__':
    test_enqueue_worker_dispatch()
    print('ok')
