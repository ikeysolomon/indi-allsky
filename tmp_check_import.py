try:
    from indi_allsky.protection_masks import _tile_median_background
    print('OK', _tile_median_background.__name__)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('ERR', e)
