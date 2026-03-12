"""Notification publisher abstraction and simple MQTT adapter.

This file provides a `NotificationPublisher` interface and an
`MQTTPublisher` that delegates to an existing `miscUpload` instance's
`mqtt_publish_message` method (keeps integration minimal).
"""
from typing import Any
import json
import logging

from .flask import db
from .flask.models import IndiAllSkyDbTaskQueueTable
from .flask.models import TaskQueueQueue, TaskQueueState
from . import constants

logger = logging.getLogger('indi_allsky')


class NotificationPublisher:
    def publish(self, topic: str, data: Any):
        """Publish data to `topic`.

        Subclasses must implement this method.
        """
        raise NotImplementedError()


class MQTTPublisher(NotificationPublisher):
    def __init__(self, misc_upload=None, upload_q=None, config=None):
        # misc_upload is optional; if provided we'll reuse its upload queue
        self.misc_upload = misc_upload
        # upload_q can be passed explicitly
        self.upload_q = upload_q or (getattr(misc_upload, 'upload_q', None) if misc_upload else None)
        self.config = config

    def publish(self, topic: str, data: Any):
        """Enqueue a MQTT upload task (no file payload) using the shared upload queue.

        This mirrors the previous behaviour where `miscUpload.mqtt_publish_message`
        created an `IndiAllSkyDbTaskQueueTable` and pushed the task id to `upload_q`.
        """
        if self.config and not self.config.get('MQTTPUBLISH', {}).get('ENABLE'):
            return

        if not self.upload_q:
            logger.warning('No upload queue available for MQTTPublisher')
            return

        try:
            jobdata = {
                'action': constants.TRANSFER_MQTT,
                'local_file': '',
                'image_topic': topic,
                'metadata': data,
                'publish_image': False,
            }

            mqtt_task = IndiAllSkyDbTaskQueueTable(
                queue=TaskQueueQueue.UPLOAD,
                state=TaskQueueState.QUEUED,
                data=jobdata,
            )
            db.session.add(mqtt_task)
            db.session.commit()

            self.upload_q.put({'task_id': mqtt_task.id})
        except Exception:
            logger.exception('Failed to enqueue mqtt publish task')
