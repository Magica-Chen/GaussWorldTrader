import importlib
import sys
import types


class UI:
    session_state = {}
    submitted = False
    errors = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __getattr__(self, name):
        if name in ('columns', 'tabs'):
            return lambda n: [self for _ in range(n if isinstance(n, int) else len(n))]
        if name == 'text_input': return lambda *a, **k: ''
        if name == 'selectbox': return lambda label, values, **k: list(values)[0]
        if name == 'form_submit_button': return lambda *a, **k: self.submitted
        if name in ('form', 'expander'): return lambda *a, **k: self
        if name == 'error': return lambda value: self.errors.append(value)
        return lambda *a, **k: None


class Client:
    def status(self):
        return {'execution_mode': 'shadow', 'data_profile': 'FREE_DELAYED', 'account_id': 'fixture'}
    def health(self): return {}
    def report(self): return {}
    def account(self): return {'equity': None}
    def plans(self): return [{'plan_id': 'fixture-plan'}]
    def command(self, *a, **k): raise AssertionError('Unauthenticated mutation')


def test_render_and_reject_unauthenticated_command(monkeypatch):
    ui = UI()
    monkeypatch.setitem(sys.modules, 'streamlit', ui)
    spec = importlib.util.spec_from_file_location('isolated_session_ui', 'src/ui/four_agent_dashboard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render_gauss_session(Client())
    ui.submitted = True
    module.render_gauss_session(Client())
    assert any('required' in error for error in ui.errors)
