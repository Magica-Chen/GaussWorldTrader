from streamlit.testing.v1 import AppTest

FIXTURE = '''
from src.ui.four_agent_dashboard import render_gauss_session
class Client:
    def status(self): return {'execution_mode':'shadow','data_profile':'FREE_DELAYED','account_id':'fixture','environment':'paper','signal_as_of':'2026-09-07T14:45:00Z','entries_paused':True,'roles':{'PostGauss':'complete','CloseGauss':'waiting','PreGauss':'waiting','LiveGauss':'manage_only'},'readiness':{'ready':False,'blockers':['fixture only']}}
    def health(self): return {'price_data':'Delayed observation: 900 seconds','news':'available','broker_stream':'fixture','reconciliation':'complete','model_workers':'idle','observed_lag_seconds':900,'feed_quality':'genuine delayed historical SIP'}
    def report(self): return {'candidates':[{'symbol':'AAPL','data_profile':'FREE_DELAYED'}], 'capital_scenarios':[{'equity':10000,'hypothetical':True,'execution_eligible':False}], 'operator_commands':[], 'suitability_reports':[{'feasible':False,'binding_limits':['actual buying power unverified']}]}
    def account(self): return {'account_id':'fixture','equity':None,'execution_eligible':False}
    def plans(self): return [{'plan_id':'fixture-plan','state':'REJECTED','reasons':['fixture only']}]
    def command(self,*args,**kwargs): raise AssertionError('No commands allowed in screenshot fixture')
render_gauss_session(Client())
'''


def test_session_renders_real_streamlit_widgets():
    app = AppTest.from_string(FIXTURE).run()
    assert not app.exception
    assert len(app.tabs) == 6
    assert any('SHADOW MODE' in x.value for x in app.info)
    assert any('HYPOTHETICAL' in x.value for x in app.warning)
    app.button[0].click().run()
    assert not app.exception
    assert any('required' in x.value for x in app.error)


def test_persisted_replay_client_renders_without_runtime_start(tmp_path):
    import json
    from pathlib import Path
    from src.runtime.service import replay_fixture

    fixture=json.loads(Path('tests/fixtures/gauss_session_day.json').read_text())
    database=tmp_path/'session.sqlite3'
    fixture['config']['database_path']=str(database)
    import uuid
    account_id='ui-fixture-'+uuid.uuid4().hex
    fixture['config']['account_id']=account_id
    fixture['account']['account_id']=account_id
    for step in fixture['steps']:
        if 'account' in step: step['account']['account_id']=account_id
    for approval in fixture.get('approvals',[]): approval['account_id']=account_id
    fixture_path=tmp_path/'fixture.json'
    fixture_path.write_text(json.dumps(fixture))
    result=replay_fixture(fixture_path, database_path=database)
    assert result['broker_writes']==0
    source=('from src.ui.four_agent_dashboard import render_gauss_session\n'
            'from src.runtime.service import SessionClient\n'
            f'render_gauss_session(SessionClient({str(database)!r}))\n')
    app=AppTest.from_string(source).run(timeout=20)
    assert not app.exception
    assert len(app.tabs)==6
    assert len(app.json)>5


def test_pricing_approval_requires_matching_verifier_before_command():
    import json
    app=AppTest.from_string(FIXTURE).run()
    next(widget for widget in app.selectbox if widget.label=='Command').select('approve_model_pricing')
    fields={widget.label:widget for widget in app.text_input}
    fields['Operator identity'].input('operator')
    fields['Control token'].input('offline-fixture-token')
    fields['Reason'].input('reviewed pricing document')
    next(widget for label,widget in fields.items() if label.startswith('For flatten')).input('fixture')
    app.text_area[0].input(json.dumps({'id':'pricing','model':'fixture','input_usd_per_million':'1',
        'output_usd_per_million':'2','verified_by':'different-operator',
        'source_reference':'fixture://pricing','expires_at':'2099-01-01T00:00:00Z'}))
    app.button[0].click().run()
    assert not app.exception
    assert any('Pricing verifier must match' in error.value for error in app.error)


def test_arming_uses_previously_displayed_scope_after_service_change():
    source='''
import streamlit as st
from src.ui.four_agent_dashboard import render_controls
class Client:
    def command(self, action, **kwargs):
        st.session_state['recorded_payload']=kwargs['payload']
        return {'state':'QUEUED'}
status={'account_id':'fixture','environment':'live','risk_policy_id':'risk-v1',
        'data_policy_version':'data-v1','deployment_version':'deployment-v1',
        'data_profile':'SUBSCRIBED_REALTIME',
        'approved_strategy_versions':[{'strategy_version':'v2' if st.session_state.get('changed') else 'v1'}]}
render_controls(Client(),status)
'''
    app=AppTest.from_string(source).run()
    app.session_state['changed']=True
    next(widget for widget in app.selectbox if widget.label=='Command').select('arm_live')
    for field in app.text_input:
        if field.label=='Operator identity':field.input('operator')
        elif field.label=='Control token':field.input('fixture-token')
        elif field.label=='Reason':field.input('reviewed version one')
        elif field.label.startswith('For flatten'):field.input('fixture')
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['recorded_payload']['approved_strategy_versions']==[{'strategy_version':'v1'}]
