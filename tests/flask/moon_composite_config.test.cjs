const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../indi_allsky/flask/templates/config.html'), 'utf8');
const script = source.slice(source.indexOf('function updateMoonCompositeControls()'),
    source.indexOf('var group_fields_camera_sqm='));

function controls(gain = '', camera = 'indi', checked = false) {
    const elements = new Map();
    const handlers = new Map();
    const document = {};
    function $(selector) {
        if (selector === document) {
            return { ready: callback => callback() };
        }
        if (!elements.has(selector)) {
            const element = {
                value: '', checked: false, disabled: false, visible: false,
                val() { return this.value; },
                prop(key, value) {
                    if (arguments.length === 1) return this[key];
                    this[key] = value;
                    return this;
                },
                toggle(visible) { this.visible = visible; return this; },
                on(events, callback) {
                    selector.split(', ').forEach(id => handlers.set(id, callback));
                    return this;
                },
            };
            elements.set(selector, element);
        }
        return elements.get(selector);
    }
    $('#CCD_CONFIG__MOONMODE__GAIN').value = gain;
    $('#CAMERA_INTERFACE').value = camera;
    $('#MOON_COMPOSITE__ENABLE').checked = checked;
    vm.runInNewContext(script, { $, document });
    return { $, change: selector => handlers.get(selector)() };
}

test('no gain hides the offer and disables all tuning controls', () => {
    const { $ } = controls();
    assert.equal($('#moon-composite-offer').visible, false);
    assert.equal($('#moon-composite-settings input').disabled, true);
    assert.equal($('#MOON_COMPOSITE__ENABLE').disabled, true);
});

test('valid gain offers the feature without enabling it, including zero gain', () => {
    for (const gain of ['0', '75', '12.5']) {
        const { $ } = controls(gain);
        assert.equal($('#moon-composite-offer').visible, true);
        assert.equal($('#MOON_COMPOSITE__ENABLE').checked, false);
        assert.equal($('#moon-composite-settings').visible, false);
        assert.equal($('#moon-composite-settings input').disabled, true);
    }
});

test('saved enabled mode restores tuning controls on load', () => {
    const { $ } = controls('75', 'indi', true);
    assert.equal($('#moon-composite-offer').visible, true);
    assert.equal($('#moon-composite-settings').visible, true);
    assert.equal($('#moon-composite-settings input').disabled, false);
});

test('typing gain and opting in reveals settings; opting out hides them', () => {
    const { $, change } = controls();
    $('#CCD_CONFIG__MOONMODE__GAIN').value = '75';
    change('#CCD_CONFIG__MOONMODE__GAIN');
    assert.equal($('#moon-composite-offer').visible, true);
    $('#MOON_COMPOSITE__ENABLE').checked = true;
    change('#MOON_COMPOSITE__ENABLE');
    assert.equal($('#moon-composite-settings input').disabled, false);
    $('#MOON_COMPOSITE__ENABLE').checked = false;
    change('#MOON_COMPOSITE__ENABLE');
    assert.equal($('#moon-composite-settings').visible, false);
    assert.equal($('#moon-composite-settings input').disabled, true);
});

test('invalid/cleared gain and unsupported interfaces hide and uncheck feature', () => {
    for (const gain of ['', ' ', '-1', 'NaN', 'Infinity', 'abc']) {
        const { $, change } = controls('75', 'indi', true);
        $('#CCD_CONFIG__MOONMODE__GAIN').value = gain;
        change('#CCD_CONFIG__MOONMODE__GAIN');
        assert.equal($('#moon-composite-offer').visible, false);
        assert.equal($('#MOON_COMPOSITE__ENABLE').checked, false);
        assert.equal($('#moon-composite-settings input').disabled, true);
    }
    const { $, change } = controls('75', 'indi', true);
    $('#CAMERA_INTERFACE').value = 'libcamera_imx477';
    change('#CAMERA_INTERFACE');
    assert.equal($('#moon-composite-offer').visible, false);
    assert.equal($('#MOON_COMPOSITE__ENABLE').checked, false);
});
