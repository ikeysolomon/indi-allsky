import importlib, sys
m = importlib.import_module('indi_allsky.protection_masks')
print('module file:', getattr(m, '__file__', None))
print('dir contains:', [n for n in dir(m) if n.startswith('_tile') or n.startswith('nebula') or n=='_tile_median_background'])
try:
    print('has _tile_median_background:', hasattr(m, '_tile_median_background'))
except Exception as e:
    print('has check error', e)
print('sys.path sample:', sys.path[:5])
