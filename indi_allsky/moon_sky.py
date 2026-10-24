"""VirtualSky alignment and ephemeris guidance for the Moon compositor."""

from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import tempfile
import time

import cv2
import ephem
import numpy

from .lens_solver import IndiAllSkyLensSolver, applySolvedValuesToConfig
from .lens_solver.calibration import displacement, pipelineSignature, validateCalibration
from .lens_solver.projection import cameraAltAz, projectToPixels
from .maskProcessing import MaskProcessor


logger = logging.getLogger('indi_allsky')


def alignment_values(config):
    sky = config.get('VIRTUALSKY', {})
    return {
        'AZIMUTH_ANGLE': config.get('LENS_AZIMUTH', 0.0),
        'LENS_ALTITUDE': config.get('LENS_ALTITUDE', 90.0),
        'POINTING_AZIMUTH': sky.get('POINTING_AZIMUTH', 0.0),
        'LATITUDE_OFFSET': sky.get('LATITUDE_OFFSET', 0.0),
        'LONGITUDE_OFFSET': sky.get('LONGITUDE_OFFSET', 0.0),
        'IMAGE_CIRCLE_DIAMETER': sky.get('IMAGE_CIRCLE_DIAMETER', 3500),
        'OFFSET_X': sky.get('OFFSET_X', 0),
        'OFFSET_Y': sky.get('OFFSET_Y', 0),
        'RADIAL_DISTORTION': sky.get('RADIAL_DISTORTION', 0.0),
        'PRECESSION': sky.get('PRECESSION', False),
    }


def has_saved_alignment(config):
    return bool(config.get('VIRTUALSKY', {}).get('ALIGNMENT'))


def image_geometry(config, sensor_shape, binning):
    processor = MaskProcessor(config)
    processor.image = numpy.zeros(sensor_shape, numpy.uint8)
    processor.binning = binning
    processor.rotate_90()
    processor.rotate_angle()
    processor.flip_v()
    processor.flip_h()
    processor.crop_image()
    if not processor.image.size:
        raise ValueError('VirtualSky image crop is empty')
    if config.get('IMAGE_SCALE', 100) != 100:
        processor.scale_image()
    processor.add_border()
    return processor.sensor_to_image, processor.image.shape


def exposure_midpoint(header, ended_at, exposure, elapsed):
    date_obs = header.get('DATE-OBS')
    if date_obs:
        date = datetime.fromisoformat(str(date_obs).replace('Z', '+00:00'))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        timestamp = date.timestamp() + exposure / 2
    else:
        if ended_at is None:
            raise ValueError('Exposure timestamp is missing')
        timestamp = float(ended_at) - elapsed + exposure / 2
        logger.debug('Moon alignment/position uses estimated exposure midpoint (no DATE-OBS)')
    if not math.isfinite(timestamp):
        raise ValueError('Exposure timestamp is invalid')
    return timestamp


class MoonSkyGuide:
    retry_seconds = 300

    def __init__(self, config):
        self.config = config
        self._retry_after = 0
        self._geometry_key = None
        self._geometry = None
        self._validated_model = None

    def is_aligned(self, i_ref, latitude=None, longitude=None):
        if not has_saved_alignment(self.config):
            return False
        saved = self.config.get('VIRTUALSKY', {}).get('ALIGNMENT', {})
        if saved.get('pipeline') != pipelineSignature(self.config):
            return False
        if saved.get('geometry', alignment_values(self.config)) != alignment_values(self.config):
            return False
        if saved.get('camera_uuid', i_ref.camera_uuid) != i_ref.camera_uuid:
            return False
        if latitude is not None and longitude is not None and saved.get('source') == 'automatic':
            if abs(saved['latitude'] - latitude) > 0.01 or abs(saved['longitude'] - longitude) > 0.01:
                return False
        shape = i_ref.hdulist[0].data.shape[-2:]
        reference_shape = saved.get('sensor_shape', shape)
        reference_binning = saved.get('binning', i_ref.binning)
        return all(a * reference_binning == b * i_ref.binning for a, b in zip(reference_shape, shape))

    def align_first_image(self, image, i_ref, latitude, longitude, persist):
        if self.is_aligned(i_ref, latitude, longitude) or time.monotonic() < self._retry_after:
            return
        self._retry_after = time.monotonic() + self.retry_seconds
        logger.info('Aligning VirtualSky from the first eligible Moon-recovery sky image')
        initial = alignment_values(self.config)
        if not has_saved_alignment(self.config):
            initial['IMAGE_CIRCLE_DIAMETER'] = int(round(
                self.config.get('LENS_IMAGE_CIRCLE', 3500) / i_ref.binning
                * self.config.get('IMAGE_SCALE', 100) / 100
            ))
        initial.update(PRECESSION=True, CALIBRATION_ENABLED=True)
        sensor_shape = i_ref.hdulist[0].data.shape[-2:]
        timestamp = exposure_midpoint(
            i_ref.hdulist[0].header, i_ref.exp_date.timestamp(), i_ref.exposure, i_ref.exp_elapsed,
        )
        # Use this exact frame, never a mutable "latest" image from another worker.
        with tempfile.TemporaryDirectory(prefix='indi-allsky-moon-solve-') as directory:
            filename = Path(directory).joinpath('sky.png')
            if not cv2.imwrite(str(filename), image):
                raise OSError('Unable to write the first-image VirtualSky solve input')
            result = IndiAllSkyLensSolver(self.config).solve(
                filename, latitude, longitude, timestamp, initial,
                lens_altitude=initial['LENS_ALTITUDE'],
                pointing_azimuth=initial['POINTING_AZIMUTH'],
                sensor_shape=sensor_shape, binning=i_ref.binning,
            )
        if not result['success'] or result.get('partial'):
            logger.warning('Automatic Moon alignment failed: %s; retrying in %d seconds',
                           result['message'], self.retry_seconds)
            return
        values = {**initial, **result['values']}
        values['CALIBRATION'] = result.get('calibration')
        if values['CALIBRATION']:
            values['CALIBRATION']['camera_uuid'] = i_ref.camera_uuid
        record = {
            'source': 'automatic', 'camera_uuid': i_ref.camera_uuid,
            'pipeline': pipelineSignature(self.config),
            'sensor_shape': list(sensor_shape), 'image_size': list(image.shape[1::-1]),
            'binning': i_ref.binning, 'latitude': latitude, 'longitude': longitude,
            'timestamp': timestamp, 'quality': result['quality'],
            'geometry': {key: values[key] for key in alignment_values(self.config)},
        }
        if not persist(values, record):
            logger.warning('Automatic Moon alignment was not saved; no new alignment applied')
            return
        applySolvedValuesToConfig(self.config, values)
        self.config['VIRTUALSKY']['ALIGNMENT'] = record
        self._retry_after = 0
        logger.info('Saved automatic VirtualSky Moon alignment: %s', result['message'])

    def search_region(self, frame, i_ref, header, latitude, longitude, elevation):
        if not self.is_aligned(i_ref, latitude, longitude):
            raise ValueError('VirtualSky alignment is unavailable or stale')
        values = alignment_values(self.config)
        saved = self.config.get('VIRTUALSKY', {}).get('ALIGNMENT', {})
        model = self.config.get('VIRTUALSKY', {}).get('CALIBRATION')
        shape = tuple(saved.get('sensor_shape', i_ref.hdulist[0].data.shape[-2:]))
        binning = saved.get('binning', i_ref.binning)
        key = (pipelineSignature(self.config), shape, binning)
        if key != self._geometry_key:
            self._geometry = image_geometry(self.config, shape, binning)
            self._geometry_key = key
        matrix, (height, width) = self._geometry
        if saved.get('image_size', [width, height]) != [width, height]:
            raise ValueError('VirtualSky alignment image geometry differs')
        timestamp = exposure_midpoint(
            header, frame.get('exp_time'), frame['exposure'], frame.get('exp_elapsed', frame['exposure']),
        )
        obs = ephem.Observer()
        obs.lat = math.radians(latitude + values['LATITUDE_OFFSET'])
        obs.lon = math.radians(longitude + values['LONGITUDE_OFFSET'])
        obs.elevation = elevation
        obs.pressure = 0
        try:
            obs.date = datetime.fromtimestamp(timestamp, timezone.utc)
        except (OverflowError, OSError) as e:
            raise ValueError('Short-exposure timestamp is outside the supported date range') from e
        moon = ephem.Moon(obs)
        alt, az = float(moon.alt), float(moon.az)
        if alt < 0 or cameraAltAz(
            alt, az, values['LENS_ALTITUDE'], values['POINTING_AZIMUTH'],
        )[0] < 0:
            raise ValueError('Predicted Moon is below the horizon or outside the lens hemisphere')
        params = [values[key] for key in (
            'AZIMUTH_ANGLE', 'LATITUDE_OFFSET', 'LONGITUDE_OFFSET',
            'IMAGE_CIRCLE_DIAMETER', 'OFFSET_X', 'OFFSET_Y', 'RADIAL_DISTORTION',
        )]
        x, y = projectToPixels(
            alt, az, params, width, height, lens_altitude=values['LENS_ALTITUDE'],
            pointing_azimuth=values['POINTING_AZIMUTH'],
        )
        if self.config.get('VIRTUALSKY', {}).get('CALIBRATION_ENABLED') and model:
            geometry = params[:6] + [values['LENS_ALTITUDE'], values['POINTING_AZIMUTH'],
                                    values['RADIAL_DISTORTION'], int(values['PRECESSION'])]
            if model is not self._validated_model:
                if not validateCalibration(model):
                    raise ValueError('Saved VirtualSky distortion calibration is invalid')
                self._validated_model = model
            model_geometry = model['geometry'] + ([0, 0] if model['version'] == 1 else [])
            if (model['pipeline'] != key[0] or model['camera_uuid'] != i_ref.camera_uuid
                    or model['image_size'] != [width, height] or model_geometry != geometry
                    or not numpy.allclose(model['context'][:2], [latitude, longitude], atol=0.01, rtol=0)):
                raise ValueError('Saved VirtualSky distortion calibration is stale')
            center = numpy.array([width / 2 + params[4], height / 2 - params[5]])
            radius = params[3] / 2
            xy = numpy.array([[x, y]])
            xy += displacement((xy - center) / radius, model) * radius
            x, y = xy[0]
        point = numpy.linalg.solve(matrix, [x, y, 1])
        scale = binning / i_ref.binning
        point[:2] = (point[:2] + 0.5) * scale - 0.5
        tolerance = max(24, params[3] * 0.05 * numpy.linalg.norm(numpy.linalg.inv(matrix[:2, :2]), ord=2) * scale)
        sensor_height, sensor_width = i_ref.hdulist[0].data.shape[-2:]
        if not numpy.isfinite(point).all() or not 0 <= point[0] < sensor_width or not 0 <= point[1] < sensor_height:
            raise ValueError('Predicted Moon is outside the sensor image')
        bounds = (
            max(0, int(point[0] - tolerance) // 2 * 2),
            max(0, int(point[1] - tolerance) // 2 * 2),
            min(sensor_width, math.ceil((point[0] + tolerance) / 2) * 2),
            min(sensor_height, math.ceil((point[1] + tolerance) / 2) * 2),
        )
        logger.debug('Predicted Moon sensor position=(%.1f, %.1f), search=%s', *point[:2], bounds)
        return bounds
