import ast
from datetime import datetime
import gzip
import io
from pathlib import Path
import select
import sys
from types import ModuleType, SimpleNamespace

import flask
import pytest
from werkzeug.exceptions import BadRequest, NotFound


class JournalReader:
    def __init__(self, entries):
        self.entries = entries
        self.groups = [[]]
        self.cursor = -1
        self.closed = False

    def add_match(self, **match):
        self.groups[-1].append(match)

    def add_disjunction(self):
        self.groups.append([])

    def matching_entries(self):
        return [
            entry for entry in self.entries if any(
                all(all(entry.get(key) == value for key, value in match.items()) for match in group)
                for group in self.groups
            )
        ]

    def seek_tail(self):
        self.cursor = len(self.matching_entries())

    def seek_head(self):
        self.cursor = -1

    def get_previous(self):
        self.cursor -= 1
        return self.matching_entries()[self.cursor] if self.cursor >= 0 else {}

    def __iter__(self):
        return self

    def __next__(self):
        entries = self.matching_entries()
        if self.cursor + 1 >= len(entries):
            raise StopIteration
        self.cursor += 1
        return entries[self.cursor]

    def fileno(self):
        return 1

    def get_events(self):
        return 1

    def process(self):
        return 1

    def close(self):
        self.closed = True


def entry(message, **source):
    return {'MESSAGE': message, **(source or {'SYSLOG_FACILITY': '22'})}


@pytest.fixture
def logs(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / 'indi_allsky' / 'flask' / 'views.py'
    names = {
        '_match_journal_sources', '_seek_journal_tail', '_journal_event',
        'StreamLogViewBase', 'StreamLogView', 'StreamIndiserverLogView', 'LogDownloadView',
    }
    nodes = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = dict(
        BaseView=object, login_required=lambda value: value, app=flask.current_app,
        Response=flask.Response, stream_with_context=flask.stream_with_context,
        send_file=flask.send_file, request=flask.request, abort=flask.abort,
        io=io, datetime=datetime,
    )
    legacy_file = tmp_path / 'capture.log'
    namespace['Path'] = lambda path: legacy_file
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    app = flask.Flask(__name__)
    app.config['ALLSKY_SERVICE_NAME'] = 'custom-capture.service'
    reader = JournalReader([])
    journal = SimpleNamespace(Reader=lambda: reader, APPEND=1)
    systemd = ModuleType('systemd')
    systemd.journal = journal
    monkeypatch.setitem(sys.modules, 'systemd', systemd)
    poller = SimpleNamespace(register=lambda *args: None, poll=lambda timeout: [])
    monkeypatch.setattr(select, 'poll', lambda: poller, raising=False)
    return SimpleNamespace(app=app, reader=reader, journal=journal, poller=poller,
                           namespace=namespace, legacy_file=legacy_file)


def download(logs, query=''):
    with logs.app.test_request_context('/log/download' + query):
        response = logs.namespace['LogDownloadView']().dispatch_request()
        response.direct_passthrough = False
        try:
            return response, gzip.decompress(response.get_data()).decode()
        finally:
            response.close()


@pytest.mark.parametrize('count', [0, 1, 9, 10, 11, 30])
def test_capture_stream_immediately_replays_last_ten_entries(logs, count):
    logs.reader.entries = [entry(str(index)) for index in range(count)]
    with logs.app.test_request_context('/stream/log'):
        response = logs.namespace['StreamLogView']().dispatch_request()
        try:
            chunks = iter(response.response)
            for index in range(max(0, count - 10), count):
                assert next(chunks) == f'data: {index}\n\n'
            assert next(chunks) == ': keep-alive\n\n'
            assert response.headers['X-Accel-Buffering'] == 'no'
            assert response.headers['Cache-Control'] == 'no-cache'
        finally:
            response.close()
    assert logs.reader.closed


def test_capture_stream_merges_unit_sources_and_legacy_facility_without_duplicates(logs):
    logs.reader.entries = [
        entry('legacy'),
        entry('system stderr', _SYSTEMD_UNIT='custom-capture.service'),
        entry('user stderr', _SYSTEMD_USER_UNIT='custom-capture.service'),
        entry('matched twice', SYSLOG_FACILITY='22', _SYSTEMD_UNIT='custom-capture.service'),
        entry('unrelated', _SYSTEMD_UNIT='other.service'),
    ]
    with logs.app.test_request_context('/stream/log'):
        response = logs.namespace['StreamLogView']().dispatch_request()
        try:
            chunks = iter(response.response)
            for message in ('legacy', 'system stderr', 'user stderr', 'matched twice'):
                assert next(chunks) == f'data: {message}\n\n'
            assert next(chunks) == ': keep-alive\n\n'
        finally:
            response.close()


def test_capture_stream_continues_after_history_and_preserves_multiline_messages(logs):
    logs.reader.entries = [entry('history')]

    def append(timeout):
        logs.reader.entries.append(entry('Moon composite\napplied=True'))
        logs.poller.poll = lambda timeout: []
        return [1]

    logs.poller.poll = append
    with logs.app.test_request_context('/stream/log'):
        response = logs.namespace['StreamLogView']().dispatch_request()
        try:
            chunks = iter(response.response)
            assert next(chunks) == 'data: history\n\n'
            assert next(chunks) == 'data: Moon composite\ndata: applied=True\n\n'
            assert next(chunks) == ': keep-alive\n\n'
        finally:
            response.close()


def test_indiserver_stream_keeps_its_separate_facility(logs):
    logs.reader.entries = [entry('capture'), entry('driver', SYSLOG_FACILITY='21')]
    with logs.app.test_request_context('/stream/indiserver'):
        response = logs.namespace['StreamIndiserverLogView']().dispatch_request()
        try:
            chunks = iter(response.response)
            assert next(chunks) == 'data: driver\n\n'
            assert next(chunks) == ': keep-alive\n\n'
        finally:
            response.close()


def test_capture_download_prefers_current_journal_over_stale_file(logs):
    logs.legacy_file.write_text('stale file', encoding='utf-8')
    logs.reader.entries = [
        entry('legacy'),
        entry('system capture', _SYSTEMD_UNIT='custom-capture.service'),
        entry('user capture', _SYSTEMD_USER_UNIT='custom-capture.service'),
    ]
    response, text = download(logs, '?lines=2')
    assert text == 'system capture\nuser capture'
    assert response.headers['X-Indi-Allsky-Log-Source'] == 'journal'
    assert logs.reader.closed


@pytest.mark.parametrize('reason', ['empty', 'permission', 'unavailable'])
def test_capture_download_preserves_explicit_legacy_file_fallback(logs, monkeypatch, caplog, reason):
    logs.legacy_file.write_bytes(b'legacy capture log\n')
    if reason == 'permission':
        def denied():
            raise PermissionError('journal denied')
        logs.journal.Reader = denied
    elif reason == 'unavailable':
        monkeypatch.setitem(sys.modules, 'systemd', None)
    response, text = download(logs)
    assert text == 'legacy capture log\n'
    assert response.headers['X-Indi-Allsky-Log-Source'] == 'file'
    assert 'legacy capture log file' in caplog.text


@pytest.mark.parametrize('value', ['0', '-1', 'bad', '1.5', ''])
def test_invalid_download_line_count_is_reported(logs, value):
    with pytest.raises(BadRequest):
        download(logs, '?lines=' + value)


def test_missing_capture_logs_are_reported(logs):
    with pytest.raises(NotFound):
        download(logs)
