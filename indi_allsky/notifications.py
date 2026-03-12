"""Central notification manager and pluggable backends.

This module centralises notification creation so future backends
like MQTT, WebPush, or Pushbullet can be plugged in without
changing call sites. It mirrors the existing DB behaviour so
changes are minimal for the PR.
"""
from datetime import datetime, timedelta
import logging

from flask import current_app as app

from . import db
from .models import IndiAllSkyDbNotificationTable

logger = logging.getLogger('indi_allsky')


class NotificationManager:
    def __init__(self, config=None):
        # keep config for future backends; not used for DB backend
        self.config = config

    def add_notification(self, category, item, notification, expire=timedelta(hours=12)):
        """Add a notification to the database.

        Behaviour intentionally matches existing `miscDb.addNotification`:
        - skip if an active identical notification exists
        - truncate notification text to 255 chars
        - commit and return the new DB row
        """
        now = datetime.now()

        notice = IndiAllSkyDbNotificationTable.query\
            .filter(IndiAllSkyDbNotificationTable.item == item)\
            .filter(IndiAllSkyDbNotificationTable.category == category)\
            .filter(IndiAllSkyDbNotificationTable.expireDate > now)\
            .first()

        if notice:
            logger.warning('Not adding existing notification')
            return

        new_notice = IndiAllSkyDbNotificationTable(
            item=item,
            category=category,
            notification=str(notification)[:255],
            expireDate=now + expire,
        )

        db.session.add(new_notice)
        db.session.commit()

        logger.info('Added %s notification: %d', category.value, new_notice.id)

        return new_notice


def add_notification(*args, **kwargs):
    """Convenience function for quick calls."""
    return NotificationManager().add_notification(*args, **kwargs)
