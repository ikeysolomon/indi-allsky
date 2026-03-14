"""Weather station runtime helper: queue API + simple process launcher.

Purpose
-------
This module consolidates two small runtime responsibilities into a single,
well-documented surface so reviewers and maintainers can reason about the
push/processing pipeline in one place:

- A lightweight, multiprocessing-backed metric queue API used by the
    Meteorologist and Push dispatcher workers (`enqueue_metric`, `get_metric`,
    `enqueue_dispatch`, `get_dispatch`, `qsizes`).
- A minimal process launcher (`main()`) that starts the meteorologist and
    push-dispatcher workers as separate processes for isolation and
    responsiveness.

Usage
-----
Import the queue helpers for in-process enqueue/get, or run `main()` to
start the worker processes. Example (project root):

        python -c "from indi_allsky import weather_station; weather_station.main({})"

"""
from multiprocessing import Queue
from typing import Any, Optional
import queue
import multiprocessing
import time
import logging

LOG = logging.getLogger('indi_allsky.weather_station')

# Multiprocessing queues used for inter-process communication.
_metric_queue: "Queue[Any]" = Queue()
_dispatch_queue: "Queue[Any]" = Queue()


# Ingestion path for Meteorologist: metrics are enqueued here and consumed by
# `indi_allsky.Meteorologist.run_worker` via `get_metric()`.
def enqueue_metric(item: Any) -> bool:
    try:
        _metric_queue.put_nowait(item)
        return True
    except Exception:
        return False


def get_metric(block: bool = True, timeout: Optional[float] = None) -> Any:
    try:
        return _metric_queue.get(block=block, timeout=timeout)
    except queue.Empty:
        raise


# Output path for PushEvaluator: dispatch items are queued here and consumed by
# `indi_allsky.push_dispatcher.run()` via `get_dispatch()`.
def enqueue_dispatch(item: Any) -> bool:
    try:
        _dispatch_queue.put_nowait(item)
        return True
    except Exception:
        return False


def get_dispatch(block: bool = True, timeout: Optional[float] = None) -> Any:
    try:
        return _dispatch_queue.get(block=block, timeout=timeout)
    except queue.Empty:
        raise


def qsizes() -> dict:
    try:
        return {"metric": _metric_queue.qsize(), "dispatch": _dispatch_queue.qsize()}
    except Exception:
        return {"metric": -1, "dispatch": -1}


def _start_process(target, name, args=()):
    p = multiprocessing.Process(target=target, name=name, args=args)
    p.daemon = False
    p.start()
    LOG.info("started %s pid=%s", name, p.pid)
    return p


def main(mqtt_cfg: dict = None):
    mqtt_cfg = dict(mqtt_cfg or {})
    logging.basicConfig(level=logging.INFO)

    # Import targets lazily so module import doesn't execute heavy code.
    from .Meteorologist import run_worker as meteorologist_run
    from .push_dispatcher import run as dispatcher_run

    procs = []
    try:
        procs.append(_start_process(meteorologist_run, 'meteorologist_worker', (mqtt_cfg, None)))
        procs.append(_start_process(dispatcher_run, 'push_dispatcher', (mqtt_cfg, None)))

        # Main thread just waits and monitors child processes.
        while True:
            alive = [p.is_alive() for p in procs]
            if not any(alive):
                LOG.info("all worker processes have exited")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        LOG.info("shutdown requested; terminating workers")
    finally:
        for p in procs:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    LOG.exception("failed to terminate %s", p.name)
        for p in procs:
            p.join(timeout=5)


if __name__ == '__main__':
    main({})
