"""Exercise the actual form, save/load snippets and conditional Moon Mode UI."""

import ast
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from jinja2 import Environment
import pytest
from wtforms import Form, BooleanField, FloatField, IntegerField, StringField
from wtforms.validators import Optional, ValidationError
from wtforms.widgets import NumberInput


ROOT = Path(__file__).resolve().parents[2]
FLASK = ROOT / 'indi_allsky' / 'flask'


def moon_form(data=None):
    tree = ast.parse((FLASK / 'forms.py').read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'IndiAllskyConfigForm')
    fields = [
        node for node in original.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and (node.targets[0].id.startswith('MOON_COMPOSITE__') or node.targets[0].id == 'CCD_CONFIG__MOONMODE__GAIN')
    ]
    validators = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ('MOON_COMPOSITE_validator', 'CCD_GAIN_validator')
    ]

    class CameraForm(Form):
        CAMERA_INTERFACE = StringField(default='indi')

    namespace = dict(
        CameraForm=CameraForm, BooleanField=BooleanField, FloatField=FloatField,
        IntegerField=IntegerField, NumberInput=NumberInput, ValidationError=ValidationError, Optional=Optional,
        math=math, __name__='indi_allsky.flask.forms', __package__='indi_allsky.flask',
    )
    cls = ast.ClassDef(name='MoonForm', bases=[ast.Name(id='CameraForm', ctx=ast.Load())],
                       keywords=[], body=fields, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=validators + [cls], type_ignores=[]))
    exec(compile(module, 'forms.py', 'exec'), namespace)
    return namespace['MoonForm'](data=data or {})


def enabled_data():
    return {
        'CAMERA_INTERFACE': 'indi', 'CCD_CONFIG__MOONMODE__GAIN': 75,
        'MOON_COMPOSITE__ENABLE': True, 'MOON_COMPOSITE__EXPOSURE': 0.001,
        'MOON_COMPOSITE__THRESHOLD': 40,
        'MOON_COMPOSITE__ROI_X1': 100, 'MOON_COMPOSITE__ROI_Y1': 200,
        'MOON_COMPOSITE__ROI_X2': 300, 'MOON_COMPOSITE__ROI_Y2': 400,
    }


@pytest.mark.parametrize('gain', [0, 75])
def test_enabled_form_accepts_valid_gain_and_region(gain):
    data = enabled_data()
    data['CCD_CONFIG__MOONMODE__GAIN'] = gain
    form = moon_form(data)
    assert form.validate(), form.errors


@pytest.mark.parametrize('key,value', [
    ('CAMERA_INTERFACE', 'indi_passive'),
    ('CCD_CONFIG__MOONMODE__GAIN', None),
    ('CCD_CONFIG__MOONMODE__GAIN', float('nan')),
    ('MOON_COMPOSITE__EXPOSURE', 0),
    ('MOON_COMPOSITE__EXPOSURE', float('nan')),
    ('MOON_COMPOSITE__EXPOSURE', ''),
    ('MOON_COMPOSITE__ROI_X1', 101),
    ('MOON_COMPOSITE__ROI_X2', 100),
    ('MOON_COMPOSITE__THRESHOLD', 250),
    ('MOON_COMPOSITE__THRESHOLD', ''),
])
def test_invalid_enabled_settings_have_field_errors(key, value):
    data = enabled_data()
    data[key] = value
    form = moon_form(data)
    assert not form.validate()
    assert form.MOON_COMPOSITE__ENABLE.errors


def test_disabled_mode_does_not_require_a_configured_region_or_native_indi():
    form = moon_form({'CAMERA_INTERFACE': 'libcamera_imx477', 'CCD_CONFIG__MOONMODE__GAIN': 0})
    assert form.validate(), form.errors
    assert not form.MOON_COMPOSITE__ENABLE.data


def test_user_can_turn_off_feature_with_incomplete_hidden_tuning():
    data = enabled_data()
    data.update(MOON_COMPOSITE__ENABLE=False, MOON_COMPOSITE__EXPOSURE='', MOON_COMPOSITE__ROI_X1='')
    form = moon_form(data)
    assert form.validate(), form.errors


def view_method(name, method):
    tree = ast.parse((FLASK / 'views.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)


def save_settings(config, form, payload):
    method = view_method('AjaxConfigView', 'dispatch_request')
    block = next(
        node for node in method.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "'MOON_COMPOSITE__ENABLE' in request.json"
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'views.py', 'exec'), {
        'self': SimpleNamespace(indi_allsky_config=config),
        'form_config': form, 'request': SimpleNamespace(json=payload),
    })


def load_settings(config):
    method = view_method('ConfigView', 'get_context')
    assignment = next(
        node for node in method.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'form_data'
    )
    pairs = [
        (key, value) for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant) and str(key.value).startswith('MOON_COMPOSITE__')
    ]
    selected = ast.Assign(targets=assignment.targets, value=ast.Dict(
        keys=[key for key, _ in pairs], values=[value for _, value in pairs],
    ))
    roi = next(
        index for index, node in enumerate(method.body) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'moon_roi'
    )
    module = ast.fix_missing_locations(ast.Module(body=[selected, *method.body[roi:roi + 2]], type_ignores=[]))
    scope = {'self': SimpleNamespace(indi_allsky_config=config)}
    exec(compile(module, 'views.py', 'exec'), scope)
    return scope['form_data']


def test_settings_save_reload_and_disable_preserve_tuning():
    data = enabled_data()
    form = moon_form(data)
    assert form.validate()
    config = {}
    save_settings(config, form, data)
    assert config['MOON_COMPOSITE'] == {
        'ENABLE': True, 'EXPOSURE': 0.001, 'ROI': [100, 200, 300, 400], 'THRESHOLD': 40,
    }
    loaded = load_settings(config)
    assert loaded == {key: value for key, value in data.items() if key.startswith('MOON_COMPOSITE__')}
    loaded['MOON_COMPOSITE__ENABLE'] = False
    disabled = moon_form({**data, **loaded})
    assert disabled.validate()
    save_settings(config, disabled, loaded)
    assert not config['MOON_COMPOSITE']['ENABLE']
    assert config['MOON_COMPOSITE']['ROI'] == [100, 200, 300, 400]
    before = config['MOON_COMPOSITE'].copy()
    save_settings(config, moon_form(), {})
    assert config['MOON_COMPOSITE'] == before


def test_missing_settings_load_safe_disabled_defaults():
    loaded = load_settings({})
    assert loaded['MOON_COMPOSITE__ENABLE'] is False
    assert loaded['MOON_COMPOSITE__EXPOSURE'] == 0.001
    assert loaded['MOON_COMPOSITE__ROI_X1'] == 0
    assert loaded['MOON_COMPOSITE__ROI_X2'] == 0


def render_controls(data=None):
    camera = (FLASK / 'templates' / 'config' / 'camera.html').read_text(encoding='utf-8')
    section = camera.split('<!-- Moon Composite -->', 1)[1].split('<!-- /Moon Composite -->', 1)[0]
    return Environment(autoescape=True).from_string(section).render(form_config=moon_form(data))


def test_rendered_controls_and_submission_registration():
    html = render_controls(enabled_data())
    template = (FLASK / 'templates' / 'config.html').read_text(encoding='utf-8')
    for field in moon_form()._fields:
        if not field.startswith('MOON_COMPOSITE__'):
            continue
        assert f'id="{field}"' in html
        assert f'id="{field}-error"' in html
        assert f"'{field}'," in template
    assert 'Only runs while Moon Mode is active' in html
    assert 'step="any"' in html


def test_conditional_moon_controls_javascript():
    node = shutil.which('node')
    assert node, 'Node.js is required to test the configuration controls'
    result = subprocess.run(
        [node, '--test', str(Path(__file__).with_name('moon_composite_config.test.cjs'))],
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
