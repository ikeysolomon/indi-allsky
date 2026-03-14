import time
from indi_allsky.rain_detection_models import compute_weighted_rain_score_with_recent_priority

def norm(val, mn, mx):
    return max(0.0, min(1.0, (val - mn) / (mx - mn) if mx != mn else 0.0))

# previous snapshot values
vals_prev = {
    'temp': norm(20.7, -30, 50),
    'hum': norm(79, 0, 100),
    'dew': norm(16.9, -30, 30),
    'pres': norm(1021.5, 900, 1100),
}
# current snapshot values from the picture overlay
vals_now = {
    'temp': norm(20.3, -30, 50),
    'hum': norm(82.1, 0, 100),
    'dew': norm(17.1, -30, 30),
    'pres': norm(1020.8, 900, 1100),
}


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

res_prev = score(vals_prev, 61)
res_now = score(vals_now, 15)
print('prev_score', res_prev['score'])
print('now_score', res_now['score'])
print('delta', res_now['score'] - res_prev['score'])
print('dew_spread_inverse', res_now['details']['dew_spread_inverse'])
