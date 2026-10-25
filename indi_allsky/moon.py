"""Experimental short-exposure moon replacement with automatic localization."""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy


logger = logging.getLogger('indi_allsky')


class MoonFootprintEdge(ValueError):
    pass


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
    margin = settings.get('SEARCH_MARGIN', 0.1)
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0.01 <= margin <= 0.1:
        raise ValueError('MOON_COMPOSITE SEARCH_MARGIN must be a fraction from 0.01 to 0.1')


def reported_exposure(header):
    return header.get('EXPTIME', header.get('EXPOSURE'))


def exposure_matches(reported, requested):
    if isinstance(reported, bool):
        return False
    try:
        value = float(reported)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(value) and value > 0 and math.isclose(
        value, requested, rel_tol=0.01, abs_tol=0.000001,
    )


def _bright_regions(gray, expected_radius=None):
    """Grow a bounded number of bright seeds, not full-frame contour searches."""
    remaining = cv2.medianBlur(gray, 3)
    height, width = gray.shape
    background = float(numpy.median(gray))
    regions = []
    for _ in range(3):
        _, peak, _, (sx, sy) = cv2.minMaxLoc(remaining)
        if peak < background + 4:
            break
        radius = 8
        while True:
            x1, y1 = max(0, sx - radius), max(0, sy - radius)
            x2, y2 = min(width, sx + radius + 1), min(height, sy + radius + 1)
            window = gray[y1:y2, x1:x2]
            threshold, _ = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contrast = peak - background
            floor = background + max(3, contrast * 0.1)
            border = numpy.concatenate((window[0], window[-1], window[:, 0], window[:, -1]))
            if numpy.median(border) > floor and (x1 > 0 or y1 > 0 or x2 < width or y2 < height):
                # Do not mistake an enclosed bright crater for the entire disk.
                radius *= 2
                continue
            threshold = max(floor, min(threshold, background + contrast * 0.35))
            binary = (window > threshold).astype(numpy.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            label = labels[sy - y1, sx - x1]
            if not label:
                remaining[sy, sx] = 0
                break
            x, y, w, h, area = (int(value) for value in stats[label])
            touches = x == 0 or y == 0 or x + w == window.shape[1] or y + h == window.shape[0]
            if touches and (x1 > 0 or y1 > 0 or x2 < width or y2 < height):
                radius *= 2
                continue
            bounds = (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
            bx1, by1, bx2, by2 = bounds
            remaining[max(0, by1 - 2):by2 + 2, max(0, bx1 - 2):bx2 + 2] = 0
            if touches or area < 6:
                break
            selected = (labels[y:y + h, x:x + w] == label).astype(numpy.uint8)
            contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            if not perimeter or contour_area < 4:
                break
            (cx, cy), disk_radius = cv2.minEnclosingCircle(contour)
            distances = numpy.linalg.norm(contour[:, 0] - (cx, cy), axis=1)
            circular_arc = numpy.mean(numpy.abs(distances - disk_radius) <= max(0.75, disk_radius * 0.12))
            circularity = 4 * math.pi * contour_area / perimeter ** 2
            occupancy = contour_area / (math.pi * disk_radius ** 2)
            # Allow a terminator on a partial disk, but require a curved limb.
            partial_disk = (
                expected_radius is not None and circularity >= 0.1
                and occupancy >= 0.08 and circular_arc >= 0.4
            )
            if not partial_disk and (circularity < 0.6 or occupancy < 0.35 or circular_arc < 0.55):
                break
            mask = numpy.zeros_like(selected)
            cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)
            brightness = float(window[y:y + h, x:x + w][selected != 0].mean())
            regions.append((bounds, mask, int(area), brightness, (bx1 + cx, by1 + cy), disk_radius))
            break
    return regions


def _spiral_labels(labels, expected):
    """Visit connected seeds in a clockwise square spiral from the prediction."""
    height, width = labels.shape
    cx = min(width - 1, max(0, round(expected[0])))
    cy = min(height - 1, max(0, round(expected[1])))
    seen = {0}
    center = int(labels[cy, cx])
    if center:
        seen.add(center)
        yield center
    for radius in range(1, max(cx, cy, width - 1 - cx, height - 1 - cy) + 1):
        forward = numpy.arange(-radius + 1, radius + 1)
        backward = numpy.arange(radius - 1, -radius - 1, -1)
        xs = cx + numpy.concatenate((
            numpy.full(2 * radius, radius), backward,
            numpy.full(2 * radius, -radius), forward,
        ))
        ys = cy + numpy.concatenate((
            forward, numpy.full(2 * radius, radius),
            backward, numpy.full(2 * radius, -radius),
        ))
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        values, first = numpy.unique(labels[ys[inside], xs[inside]], return_index=True)
        for index in numpy.argsort(first):
            label = int(values[index])
            if label not in seen:
                seen.add(label)
                yield label


def _guided_regions(gray, expected, expected_radius):
    """Grow spiral seeds to their complete bounded, not necessarily round footprint."""
    background = float(numpy.median(gray))
    peak = int(cv2.medianBlur(gray, 3).max())
    contrast = peak - background
    if contrast < 4:
        logger.warning(
            'Moon search has insufficient contrast: peak=%d background=%.1f contrast=%.1f; '
            'check predicted position, clouds and short exposure',
            peak, background, contrast,
        )
        return []
    threshold = background + max(3, contrast * 0.1)
    bright = (gray > threshold).astype(numpy.uint8)
    # Bridge small cloud interruptions, not a predicted circular disk.
    gap = max(1, math.ceil(expected_radius * 0.15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    connected = cv2.morphologyEx(
        numpy.pad(bright, gap), cv2.MORPH_CLOSE, kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )[gap:-gap, gap:-gap]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(connected)
    regions = []
    for label in _spiral_labels(labels, expected):
        x, y, width, height, _ = (int(value) for value in stats[label])
        if x == 0 or y == 0 or x + width == gray.shape[1] or y + height == gray.shape[0]:
            continue
        if max(width, height) > expected_radius * 3.6 + 2:
            continue
        component = (labels[y:y + height, x:x + width] == label).astype(numpy.uint8)
        observed = (component != 0) & (bright[y:y + height, x:x + width] != 0)
        area = int(numpy.count_nonzero(observed))
        if area < max(6, math.pi * expected_radius ** 2 * 0.08):
            continue
        if cv2.distanceTransform(observed.astype(numpy.uint8), cv2.DIST_L2, 5).max() <= 2:
            continue
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not 0.55 * expected_radius <= radius <= 1.8 * expected_radius:
            continue
        silhouette = numpy.zeros_like(component)
        cv2.drawContours(silhouette, [contour], -1, 1, cv2.FILLED)
        brightness = float(gray[y:y + height, x:x + width][observed].mean())
        regions.append(((x, y, x + width, y + height), silhouette, area, brightness, (x + cx, y + cy), radius))
    logger.info(
        'Moon spiral search: prediction=(%.1f, %.1f) radius=%.1f window=%dx%d '
        'peak=%d background=%.1f threshold=%.1f candidates=%d',
        *expected, expected_radius, gray.shape[1], gray.shape[0], peak, background, threshold, len(regions),
    )
    return regions


def _select_region(gray, expected=None, expected_radius=None):
    guided = expected is not None and expected_radius is not None
    regions = _guided_regions(gray, expected, expected_radius) if guided else _bright_regions(gray, expected_radius)
    if not regions:
        if int(gray.max()) - float(numpy.median(gray)) < 4:
            logger.warning('Moon may be too faint or prediction misaligned; check alignment before increasing manual EXPOSURE')
        else:
            logger.warning('Moon composite skipped: no complete resolved lunar footprint within the search bounds'
                           if guided else 'Moon composite skipped: no resolved circular lunar candidate (small source or non-lunar shape)')
        return None
    if expected_radius is not None:
        regions = [region for region in regions if 0.55 * expected_radius <= region[5] <= 1.8 * expected_radius]
        if not regions:
            logger.warning('Moon composite skipped: candidate size differs from predicted lunar radius; rejecting point sources')
            return None
    if expected is not None:
        def distance(region):
            return math.hypot(region[4][0] - expected[0], region[4][1] - expected[1])

        if not guided:
            regions.sort(key=lambda region: (distance(region), -region[2] * region[3]))
        best = regions[0]
        if any(
            abs(distance(region) - distance(best)) <= max(2, best[5] * 0.5)
            and region[2] >= best[2] * 0.1 and region[3] >= best[3] * 0.5
            for region in regions[1:]
        ):
            logger.warning('Moon composite skipped: ambiguous nearby lunar candidates')
            return None
        return best
    # Integrated brightness favors a resolved lunar disk over isolated hot pixels.
    regions.sort(key=lambda region: region[2] * region[3], reverse=True)
    best = regions[0]
    if any(region[2] >= best[2] * 0.1 and region[3] >= best[3] * 0.5 for region in regions[1:]):
        logger.warning('Moon composite skipped: ambiguous circular targets')
        return None
    return best


def find_moon(sensor, bayer=False, search=None, expected=None, expected_radius=None):
    """Locate on a max-pooled sensor preview; return an even, padded sensor ROI."""
    if sensor.dtype != numpy.uint8 or sensor.ndim not in (2, 3):
        raise ValueError('Moon detection requires 8-bit mono or channel-first sensor data')
    offset_x = offset_y = 0
    if search is not None:
        offset_x, offset_y, right, bottom = search
        height, width = sensor.shape[-2:]
        if not 0 <= offset_x < right <= width or not 0 <= offset_y < bottom <= height:
            raise ValueError('Predicted Moon search region is outside the sensor image')
        if bayer and (offset_x % 2 or offset_y % 2):
            raise ValueError('Predicted Moon search origin must preserve Bayer phase')
        sensor = sensor[..., offset_y:bottom, offset_x:right]
    gray = sensor.max(axis=0) if sensor.ndim == 3 else sensor
    height, width = gray.shape
    step = max(1, math.ceil(max(height, width) / 768))
    if bayer:
        step = max(2, step + step % 2)
    # Top-left anchoring makes each sampled dilation exactly one max-pool block.
    preview = cv2.dilate(
        gray, numpy.ones((step, step), numpy.uint8), anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )[::step, ::step]
    preview_expected = None if expected is None else (
        (expected[0] - offset_x) / step, (expected[1] - offset_y) / step,
    )
    region = _select_region(
        preview, preview_expected, None if expected_radius is None else expected_radius / step,
    )
    if region is None:
        return None
    (x1, y1, x2, y2), _, _, _, _, _ = region
    margin = max(6, math.ceil(max(x2 - x1, y2 - y1) * step * 0.25))
    roi = (
        offset_x + max(0, (x1 * step - margin) // 2 * 2),
        offset_y + max(0, (y1 * step - margin) // 2 * 2),
        offset_x + min(width, math.ceil((x2 * step + margin) / 2) * 2),
        offset_y + min(height, math.ceil((y2 * step + margin) / 2) * 2),
    )
    logger.debug('Moon automatically localized: sensor ROI=%s preview step=%d', roi, step)
    return roi


def replacement_region(long_sensor, short_sensor, roi, bounded=False):
    """Expand a short-frame crop to the connected bright long-frame footprint."""
    long_gray = long_sensor.max(axis=0) if long_sensor.ndim == 3 else long_sensor
    short_gray = short_sensor.max(axis=0) if short_sensor.ndim == 3 else short_sensor
    if long_gray.shape != short_gray.shape or long_gray.dtype != numpy.uint8 or short_gray.dtype != numpy.uint8:
        raise ValueError('Moon footprint requires matching linear 8-bit sensor regions')
    x1, y1, x2, y2 = roi
    if not bounded:
        margin = max(24, 2 * max(x2 - x1, y2 - y1))
        left, top = max(0, x1 - margin), max(0, y1 - margin)
        right, bottom = min(long_gray.shape[1], x2 + margin), min(long_gray.shape[0], y2 + margin)
        region, mask = replacement_region(
            long_gray[top:bottom, left:right], short_gray[top:bottom, left:right],
            (x1 - left, y1 - top, x2 - left, y2 - top), bounded=True,
        )
        return (region[0] + left, region[1] + top, region[2] + left, region[3] + top), mask
    short_roi = short_gray[y1:y2, x1:x2]
    background = float(numpy.median(short_roi))
    threshold = background + max(3, (int(short_roi.max()) - background) * 0.1)
    seeds = short_roi > threshold
    if not seeds.any():
        return roi, None
    long_background = float(numpy.median(long_gray))
    peak = float(numpy.percentile(long_gray[y1:y2, x1:x2][seeds], 90))
    if peak >= 250 and long_background >= peak - 3:
        raise MoonFootprintEdge('bright lunar footprint reaches the bounded search edge')
    # A relative plateau threshold also works after a dark frame lowers clipping levels.
    level = long_background + max(3, (peak - long_background) * 0.95)
    saturated = cv2.morphologyEx(
        (long_gray > level).astype(numpy.uint8), cv2.MORPH_CLOSE, numpy.ones((3, 3), numpy.uint8),
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(saturated)
    overlaps = numpy.bincount(labels[y1:y2, x1:x2][seeds], minlength=len(stats))
    overlaps[0] = 0
    label = int(overlaps.argmax())
    if overlaps[label] < max(3, int(numpy.count_nonzero(seeds) * 0.1)):
        return roi, None
    x, y, width, height, area = (int(value) for value in stats[label])
    if area <= numpy.count_nonzero(seeds) * 1.05:
        return roi, None
    margin = max(8, math.ceil(max(width, height) * 0.08))
    left, top = min(x1, (x - margin) // 2 * 2), min(y1, (y - margin) // 2 * 2)
    right = max(x2, math.ceil((x + width + margin) / 2) * 2)
    bottom = max(y2, math.ceil((y + height + margin) / 2) * 2)
    if left < 0 or top < 0 or right > long_gray.shape[1] or bottom > long_gray.shape[0]:
        raise MoonFootprintEdge('bright lunar footprint reaches the bounded search edge')
    mask = (labels[top:bottom, left:right] == label).astype(numpy.uint8)
    logger.info('Moon replacement mask follows long footprint: %dx%d; short Moon is not resized', width, height)
    return (left, top, right, bottom), mask


def _sky_background(target, ring):
    yy, xx = numpy.indices(ring.shape, dtype=numpy.float32)
    xx /= max(1, ring.shape[1] - 1)
    yy /= max(1, ring.shape[0] - 1)
    design = numpy.column_stack((numpy.ones(numpy.count_nonzero(ring), numpy.float32), xx[ring], yy[ring]))
    values = target[ring].astype(numpy.float32)
    try:
        coefficients, _, rank, _ = numpy.linalg.lstsq(design, values, rcond=None)
        residual = numpy.abs(values - design @ coefficients)
        if residual.ndim == 2:
            residual = residual.max(axis=1)
        good = residual <= max(2, float(numpy.median(residual)) * 3)
        if numpy.count_nonzero(good) < 16:
            raise ValueError('Insufficient reliable sky samples around the Moon footprint')
        coefficients, _, rank, _ = numpy.linalg.lstsq(design[good], values[good], rcond=None)
    except numpy.linalg.LinAlgError as e:
        raise ValueError('Unable to fit sky illumination around the Moon footprint') from e
    if rank != 3:
        raise ValueError('Moon background samples do not constrain a spatial sky match')
    if target.ndim == 3:
        xx, yy = xx[:, :, None], yy[:, :, None]
    return numpy.clip(coefficients[0] + coefficients[1] * xx + coefficients[2] * yy, 0, 255)


def _moon_display(patch, display_patch, detail_patch, target, mask, selected, force_match, radius):
    ring = cv2.dilate(mask, numpy.ones((7, 7), numpy.uint8)) != mask
    if numpy.count_nonzero(ring) < 16:
        raise ValueError('Insufficient surrounding sky to match the lunar replacement background')
    detail = patch if detail_patch is None else detail_patch
    if detail.shape != patch.shape or detail.dtype.kind not in 'ui':
        raise ValueError('Native Moon detail must match the linear ROI geometry and integer format')
    native_peak = detail.max(axis=2) if detail.ndim == 3 else detail
    display_peak = display_patch.max(axis=2) if display_patch.ndim == 3 else display_patch
    native_low, native_high = numpy.percentile(native_peak[selected], (5, 95))
    display_low, display_high = numpy.percentile(display_peak[selected], (5, 95))
    sky_peak = target.max(axis=2) if target.ndim == 3 else target
    sky_level = float(numpy.median(sky_peak[ring]))
    lost_detail = (
        display_low <= sky_level + 4
        or (native_high > native_low and display_high - display_low < 8)
        or (native_high > native_low and numpy.mean(display_peak[selected] >= 250) > 0.5)
    )
    if not force_match and not lost_detail:
        return display_patch
    source = detail if lost_detail else display_patch
    if lost_detail:
        logger.warning(
            'Moon sky transfer loses lunar contrast: native %.1f..%.1f, display %.1f..%.1f, sky %.1f; using native short-frame detail',
            native_low, native_high, display_low, display_high, sky_level,
        )
    source = source.astype(numpy.float32)
    source -= numpy.median(source[ring], axis=0).astype(numpy.float32)
    source_peak = source.max(axis=2) if source.ndim == 3 else source
    white = float(numpy.percentile(source_peak[selected], 99))
    if white <= 0:
        raise ValueError('Short Moon has no usable contrast above its background')
    background = _sky_background(target, ring)
    background_peak = background.max(axis=2) if background.ndim == 3 else background
    if numpy.median(background_peak[selected]) >= 245:
        raise ValueError('Sky around the Moon is too bright to display lunar contrast reliably')
    normalized = numpy.clip(source / white, 0, 1)
    display_white = 250 if lost_detail else float(numpy.percentile(display_peak[selected], 99))
    logger.info('Moon display matched to spatial sky background; lunar radius=%.2fpx, native detail=%s', radius, lost_detail)
    return numpy.clip(background + normalized * (display_white - background), 0, 255)


def blend(image, patch, roi, display_patch=None, expected=None, expected_radius=None, replacement_mask=None, detail_patch=None, preserve_transfer=False):
    """Blend same-scale short pixels; optionally include the blown-out long footprint."""
    x1, y1, x2, y2 = roi
    if image.dtype != numpy.uint8 or patch.dtype != numpy.uint8:
        raise ValueError('Moon composite requires 8-bit display data')
    if (
        x2 > image.shape[1] or y2 > image.shape[0]
        or patch.shape != image[y1:y2, x1:x2].shape
    ):
        raise ValueError('Moon composite ROI does not match image geometry')
    if display_patch is None:
        display_patch = patch
    if display_patch.shape != patch.shape or display_patch.dtype != numpy.uint8:
        raise ValueError('Moon display patch must match the linear ROI geometry and 8-bit format')

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    patch_expected = None if expected is None else (expected[0] - x1, expected[1] - y1)
    region = _select_region(gray, patch_expected, expected_radius)
    if region is None:
        return False
    (x, y, right, bottom), silhouette, area, _, _, radius = region
    if area < 16:
        logger.warning('Moon too small to resolve surface detail; keeping the normal image')
        return False
    dilation = min(3, max(1, round(radius * 0.02)))
    feather = min(4, max(1, round(radius * 0.04)))
    padding = max(4, dilation + feather + 1)
    if x < padding or y < padding or right > gray.shape[1] - padding or bottom > gray.shape[0] - padding:
        logger.warning('Moon composite skipped: target touches ROI edge')
        return False
    mask = numpy.zeros_like(gray)
    mask[y:bottom, x:right] = silhouette
    lunar_mask = mask
    selected = mask != 0
    peak = patch.max(axis=2) if patch.ndim == 3 else patch
    if numpy.count_nonzero(peak[selected] >= 250) > area * 0.01:
        logger.warning('Moon composite skipped: short exposure is clipped; reduce EXPOSURE')
        return False

    target = image[y1:y2, x1:x2]
    if replacement_mask is not None:
        if replacement_mask.shape != gray.shape or replacement_mask.dtype != numpy.uint8 or replacement_mask.max() > 1:
            raise ValueError('Moon replacement mask must match the ROI and contain only zero or one')
        bloom_radius = math.sqrt(numpy.count_nonzero(replacement_mask) / math.pi)
        bloom_feather = min(16, max(2, round(bloom_radius * 0.075)))
        padding = max(padding, 2 * bloom_feather)
        if (replacement_mask[:padding].any() or replacement_mask[-padding:].any()
                or replacement_mask[:, :padding].any() or replacement_mask[:, -padding:].any()):
            raise ValueError('Moon replacement footprint touches ROI edge')
        mask = numpy.maximum(mask, replacement_mask)
    if not preserve_transfer and (replacement_mask is not None or detail_patch is not None):
        display_patch = _moon_display(
            patch, display_patch, detail_patch, target, mask, selected, replacement_mask is not None, radius,
        )

    lunar_mask = cv2.dilate(lunar_mask, numpy.ones((2 * dilation + 1, 2 * dilation + 1), numpy.uint8))
    alpha = cv2.GaussianBlur(lunar_mask.astype(numpy.float32), (2 * feather + 1, 2 * feather + 1), 0)
    if replacement_mask is not None:
        kernel = numpy.ones((2 * bloom_feather + 1, 2 * bloom_feather + 1), numpy.uint8)
        # Fade outside the footprint, never blend saturated pixels back into it.
        bloom_mask = cv2.dilate(replacement_mask, kernel)
        bloom_alpha = cv2.GaussianBlur(
            bloom_mask.astype(numpy.float32), kernel.shape, 0,
        )
        numpy.maximum(alpha, bloom_alpha, out=alpha)
        logger.info('Moon footprint feather=%dpx; lunar limb feather=%dpx', bloom_feather, feather)
    if patch.ndim == 3:
        alpha = alpha[:, :, None]
    weighted_patch = numpy.multiply(display_patch, alpha)
    numpy.subtract(1.0, alpha, out=alpha)
    blended = numpy.multiply(target, alpha)
    numpy.add(blended, weighted_patch, out=blended)
    numpy.rint(blended, out=blended)
    target[:] = blended
    return True
