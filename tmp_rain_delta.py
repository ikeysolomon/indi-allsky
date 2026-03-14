import time
from indi_allsky.rain_detection_models import compute_weighted_rain_score_with_recent_priority

def norm(val, mn, mx):
    return max(0.0, min(1.0, (val - mn) / (mx - mn) if mx != mn else 0.0))


def score(vals, stars):
    now = int(time.time())
    ts1 = now - 1800
    ts2 = now
    hist = {
        'temperature': [(ts1, vals['temp']), (ts2, vals['temp'])],
        'humidity': [(ts1, vals['hum']), (ts2, vals['hum'])],
        'dew_point': [(ts1, vals['dew']), (ts2, vals['dew'])],
        'pressure': [(ts1, vals['pres']), (ts2, vals['pres'])],
    }
    high_cloud = max(0.0, min(1.0, (100 - stars) / 100.0))
    return compute_weighted_rain_score_with_recent_priority(hist, high_cloud=high_cloud)

vals1 = {
    'temp': norm(20.7, -30, 50),
    'hum': norm(79, 0, 100),
    'dew': norm(16.9, -30, 30),
    'pres': norm(1021.5, 900, 1100),
}
vals2 = {
    'temp': norm(20.3, -30, 50),
    'hum': norm(84.8, 0, 100),
    'dew': norm(17.6, -30, 30),
    'pres': norm(1021.2, 900, 1100),
}
res1 = score(vals1, 61)
res2 = score(vals2, 36)
print('earlier', res1['score'])
print('current', res2['score'])
print('delta', res2['score'] - res1['score'])
