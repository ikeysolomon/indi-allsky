"""Experimental short-exposure moon replacement, limited to a sensor ROI."""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy


logger = logging.getLogger('indi_allsky')


@dataclass
class MoonCapture:
    id: int
    long_request: Optional[Tuple[float, float, int]] = None
    phase: str = 'normal'
    status: str = 'prepared'
    short_frame: Optional[dict] = None
    failure: Optional[str] = None


def validate_config(config):
    settings = config.get('MOON_COMPOSITE', {})
    if not settings.get('ENABLE'):
        return
    if config.get('CAMERA_INTERFACE') != 'indi':
        raise ValueError('MOON_COMPOSITE requires the native indi camera interface')
    exposure = settings.get('EXPOSURE', 0.001)
    if type(exposure) not in (int, float) or not math.isfinite(exposure) or exposure <= 0:
        raise ValueError('MOON_COMPOSITE EXPOSURE must be finite and positive')
    roi = settings.get('ROI', [])
    if (
        not isinstance(roi, (list, tuple)) or len(roi) != 4
        or any(type(value) is not int or value < 0 for value in roi)
        or roi[2] <= roi[0]
        or roi[3] <= roi[1]
        or any(value % 2 for value in roi)
    ):
        raise ValueError('MOON_COMPOSITE ROI requires even sensor coordinates [x1, y1, x2, y2]')
    threshold = settings.get('THRESHOLD', 40)
    if type(threshold) is not int or not 1 <= threshold < 250:
        raise ValueError('MOON_COMPOSITE THRESHOLD must be an integer from 1 to 249')


def blend(image, patch, roi, threshold=40):
    """Replace only the largest isolated lunar silhouette; leave sky untouched."""
    x1, y1, x2, y2 = roi
    if image.dtype != numpy.uint8 or patch.dtype != numpy.uint8:
        raise ValueError('Moon composite requires 8-bit display data')
    if (
        x2 > image.shape[1] or y2 > image.shape[0]
        or patch.shape != image[y1:y2, x1:x2].shape
    ):
        raise ValueError('Moon composite ROI does not match image geometry')

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    mask = (gray >= threshold).astype(numpy.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count == 1:
        logger.warning('Moon composite skipped: no moon in ROI')
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(numpy.argmax(areas)) + 1
    area = int(areas[largest - 1])
    if area < 16 or numpy.count_nonzero(areas >= max(16, area * 0.1)) > 1:
        logger.warning('Moon composite skipped: small or ambiguous target')
        return False
    x, y, width, height, _ = stats[largest]
    if x < 4 or y < 4 or x + width > gray.shape[1] - 4 or y + height > gray.shape[0] - 4:
        logger.warning('Moon composite skipped: target touches ROI edge')
        return False
    selected = labels == largest
    peak = patch.max(axis=2) if patch.ndim == 3 else patch
    if numpy.count_nonzero(peak[selected] >= 250) > area * 0.01:
        logger.warning('Moon composite skipped: short exposure is clipped; reduce EXPOSURE')
        return False

    mask = selected.astype(numpy.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    mask = cv2.dilate(mask, numpy.ones((3, 3), numpy.uint8))
    alpha = cv2.GaussianBlur(mask.astype(numpy.float32), (5, 5), 0)
    if patch.ndim == 3:
        alpha = alpha[:, :, None]
    target = image[y1:y2, x1:x2]
    target[:] = numpy.rint(target * (1.0 - alpha) + patch * alpha).astype(numpy.uint8)
    return True
