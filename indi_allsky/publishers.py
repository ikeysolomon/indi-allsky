"""Notification publisher abstraction and simple MQTT adapter.

This file provides a `NotificationPublisher` interface and an
`MQTTPublisher` that delegates to an existing `miscUpload` instance's
`mqtt_publish_message` method (keeps integration minimal).
"""
from typing import Any


class NotificationPublisher:
    def publish(self, topic: str, data: Any):
        """Publish data to `topic`.

        Subclasses must implement this method.
        """
        raise NotImplementedError()


class MQTTPublisher(NotificationPublisher):
    def __init__(self, misc_upload):
        # misc_upload is expected to implement `mqtt_publish_message(topic, mq_data)`
        self.misc_upload = misc_upload

    def publish(self, topic: str, data: Any):
        if not self.misc_upload:
            return
        try:
            self.misc_upload.mqtt_publish_message(topic, data)
        except Exception:
            # keep publisher failures non-fatal for callers
            pass
