import logging

logger = logging.getLogger('indi_allsky')


class IndiAllSky_Stretch_Base(object):
    def __init__(self, *args, **kwargs):
        self.config = args[0]
        self.reset_display_transfer()


    def reset_display_transfer(self):
        self._display_luts = ()
        self.record_display_transfer = False


    def apply_last_stretch(self, data):
        if not self._display_luts:
            raise ValueError('No completed display stretch is available for the Moon ROI')
        for lut in self._display_luts:
            data = lut.take(data, mode='raise')
        return data
