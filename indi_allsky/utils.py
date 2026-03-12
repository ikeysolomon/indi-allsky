"""Small reusable utilities for push evaluation and history analysis.

These helpers operate on the `history` sequence used by PushEvaluator and
other components. Keeping them here makes them testable and reusable.
"""
from typing import List, Tuple, Optional
import numpy as np

def get_series(history: List[Tuple[float, dict]], slot: str, window_s: Optional[float] = None, now_ts: Optional[float] = None):
    """Return a list of (ts, value) tuples for numeric values of `slot` in history.

    history is expected to be an iterable of (ts, payload) as produced by
    `PushHistoryCache.get_recent()` or similar. If `window_s` is provided the
    returned series will include only entries with ts >= now - window_s.
    """
    if now_ts is None:
        now_ts = __import__('time').time()
    cutoff_ts = now_ts - window_s if window_s else 0
    series = []
    for ts, d in history:
        if ts < cutoff_ts:
            continue
        if slot in d and isinstance(d[slot], (int, float)):
            series.append((ts, float(d[slot])))
    return series

def trend_slope(history: List[Tuple[float, dict]], slot: str, window_s: Optional[float] = None):
    """Estimate trend slope (per minute) for a numeric slot in the history.

    Returns a float slope (units per minute). If insufficient data or an
    error occurs, returns 0.0.
    """
    s = get_series(history, slot, window_s)
    if len(s) < 2:
        return 0.0
    xs = np.array([int(t) for t, v in s], dtype=float)
    ys = np.array([v for t, v in s], dtype=float)
    xs0 = xs - xs[0]
    try:
        p = np.polyfit(xs0, ys, 1)
        slope_per_second = float(p[0])
        slope = slope_per_second * 60.0
    except Exception:
        slope = 0.0
    return slope

def increasing(history: List[Tuple[float, dict]], slot: str, window_s: Optional[float] = None, min_slope: float = 0.0):
    return trend_slope(history, slot, window_s) > float(min_slope)

def decreasing(history: List[Tuple[float, dict]], slot: str, window_s: Optional[float] = None, min_slope: float = 0.0):
    return trend_slope(history, slot, window_s) < -float(min_slope)import math
from datetime import datetime
from datetime import timedelta
import logging

import ephem

from . import constants

logger = logging.getLogger('indi_allsky')


class IndiAllSkyDateCalcs(object):

    def __init__(self, config, position_av):
        self.config = config

        self.position_av = position_av

        self.night_sun_radians = math.radians(self.config['NIGHT_SUN_ALT_DEG'])


    def calcDayDate(self, now):
        utc_offset = now.astimezone().utcoffset()

        utcnow_notz = now - utc_offset

        obs = ephem.Observer()
        sun = ephem.Sun()
        obs.lon = math.radians(self.position_av[constants.POSITION_LONGITUDE])
        obs.lat = math.radians(self.position_av[constants.POSITION_LATITUDE])
        obs.elevation = self.position_av[constants.POSITION_ELEVATION]

        # disable atmospheric refraction calcs
        obs.pressure = 0

        obs.date = utcnow_notz
        sun.compute(obs)
        night = sun.alt < self.night_sun_radians


        start_day = datetime.strptime(now.strftime('%Y%m%d'), '%Y%m%d')
        start_day_utc = start_day - utc_offset

        obs.date = start_day_utc
        sun.compute(obs)


        today_meridian = obs.next_transit(sun).datetime()
        obs.date = today_meridian
        sun.compute(obs)

        previous_antimeridian = obs.previous_antitransit(sun).datetime()
        next_antimeridian = obs.next_antitransit(sun).datetime()

        obs.date = next_antimeridian
        sun.compute(obs)


        if utcnow_notz < previous_antimeridian:
            #logger.warning('Pre-antimeridian')
            dayDate = (now - timedelta(days=1)).date()
        elif utcnow_notz < today_meridian:
            #logger.warning('Pre-meridian')

            if night:
                dayDate = (now - timedelta(days=1)).date()
            else:
                dayDate = now.date()
        elif utcnow_notz < next_antimeridian:
            #logger.warning('Post-meridian')
            dayDate = now.date()
        else:
            #logger.warning('Post-antimeridian')

            if night:
                dayDate = now.date()
            else:
                dayDate = (now + timedelta(days=1)).date()


        return dayDate


    def getDayDate(self):
        now = datetime.now()
        return self.calcDayDate(now)


    def getNextDayNightTransition(self):
        now = datetime.now()
        utc_offset = now.astimezone().utcoffset()
        utcnow_notz = now - utc_offset


        obs = ephem.Observer()
        sun = ephem.Sun()
        obs.lon = math.radians(self.position_av[constants.POSITION_LONGITUDE])
        obs.lat = math.radians(self.position_av[constants.POSITION_LATITUDE])
        obs.elevation = self.position_av[constants.POSITION_ELEVATION]

        # disable atmospheric refraction calcs
        obs.pressure = 0

        obs.date = utcnow_notz
        sun.compute(obs)
        night = sun.alt < self.night_sun_radians


        start_day = datetime.strptime(now.strftime('%Y%m%d'), '%Y%m%d')
        start_day_utc = start_day - utc_offset

        obs.date = start_day_utc
        sun.compute(obs)


        today_meridian = obs.next_transit(sun).datetime()
        obs.date = today_meridian
        sun.compute(obs)

        next_meridian = obs.next_transit(sun).datetime()
        previous_antimeridian = obs.previous_antitransit(sun).datetime()
        next_antimeridian = obs.next_antitransit(sun).datetime()

        obs.date = next_antimeridian
        sun.compute(obs)
        next_antimeridian_2 = obs.next_antitransit(sun).datetime()


        if utcnow_notz < previous_antimeridian:
            #logger.warning('Pre-antimeridian')
            night_stop = today_meridian

            if night:
                day_stop = next_antimeridian
            else:
                day_stop = previous_antimeridian
        elif utcnow_notz < today_meridian:
            #logger.warning('Pre-meridian')

            if night:
                night_stop = today_meridian
            else:
                night_stop = next_meridian

            day_stop = next_antimeridian
        elif utcnow_notz < next_antimeridian:
            #logger.warning('Post-meridian')
            night_stop = next_meridian

            if night:
                day_stop = next_antimeridian_2
            else:
                day_stop = next_antimeridian
        else:
            #logger.warning('Post-antimeridian')
            night_stop = next_meridian
            day_stop = next_antimeridian_2


        if night_stop < day_stop:
            next_stop = night_stop
        else:
            next_stop = day_stop


        return next_stop + utc_offset
