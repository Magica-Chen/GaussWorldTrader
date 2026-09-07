"""Real frontend tab switching with immutable actual-account execution state (T59)."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from src.runtime.models import AccountProfile, RuntimeConfig
from src.runtime.service import SessionClient
from src.runtime.store import Store


def test_t59_interactive_scenario_switch_preserves_actual_account_and_reservations(tmp_path):
    browser_deps = Path('/tmp/gauss-browser-deps')
    browsers = list((Path.home()/'.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-linux/headless_shell'))
    if not browser_deps.is_dir() or not browsers:
        pytest.skip('Local T59 browser fixture requires isolated Playwright and cached Chromium')
    now = datetime(2026, 9, 9, 15, tzinfo=timezone.utc)
    account_id = 't59-actual-' + uuid4().hex
    database = tmp_path/'state.sqlite3'
    config = RuntimeConfig(database_path=str(database), account_id=account_id)
    account = AccountProfile(account_id=account_id, environment='paper', hypothetical=False,
                             equity='10000', cash='8000', buying_power='8000', currency='USD',
                             reserved_capital='250', observed_at=now, mandate_id=config.policy.id)
    scope = 'paper:'+account_id
    store = Store(database)
    store.project('configuration',scope,config.model_dump(mode='json'),scope)
    store.project('account',scope,account.model_dump(mode='json'),scope)
    store.project('runtime',scope,{'scope':scope,'account_id':account_id,'environment':'paper',
        'execution_mode':'shadow','data_profile':'FREE_DELAYED','roles':{},'entries_paused':True,
        'supervision':'NOT_RUNNING','signal_as_of':'2026-09-09T14:45:00+00:00',
        'updated_at':now.isoformat(),'watermarks':{'TEST':'2026-09-09T14:44:00+00:00'}},scope)
    with store.transaction():
        store.db.execute('INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?)',
                         ('t59-reservation',scope,'fixture-session','250','5','TEST','RESERVED',0))
    store.close()

    def execution_state():
        with sqlite3.connect(database) as db:
            reservations = db.execute('SELECT * FROM reservations ORDER BY intent_id').fetchall()
        return SessionClient(database,account_id=account_id,environment='paper').account(),reservations

    baseline = execution_state()
    scenarios = [AccountProfile.model_validate({**account.model_dump(), 'id':uuid4().hex,'account_id':f't59-scenario-{capital}',
        'environment':'scenario','hypothetical':True,'equity':capital,'cash':capital,
        'buying_power':capital,'reserved_capital':'0'}).model_dump(mode='json')
        for capital in ['5000','250000']]
    comparison = SessionClient(database,account_id=account_id,environment='paper').scenarios(
        scenarios,data_profile='SUBSCRIBED_REALTIME')
    assert comparison['execution_eligible'] is False
    assert comparison['data_profile']=='SUBSCRIBED_REALTIME'
    assert all(row['hypothetical'] for row in comparison['reports'])
    assert execution_state()==baseline

    app=tmp_path/'app.py'
    app.write_text('from src.runtime.service import SessionClient\n'
                   'from src.ui.four_agent_dashboard import render_gauss_session\n'
                   f'render_gauss_session(SessionClient({str(database)!r},account_id={account_id!r},environment="paper"))\n')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path.cwd())+':/tmp/gauss-test-deps'
    log=(tmp_path/'streamlit.log').open('w')
    server=subprocess.Popen([sys.executable,'-m','streamlit','run',str(app),
        '--global.developmentMode=false','--server.headless=true',f'--server.port={port}',
        '--server.address=127.0.0.1','--browser.gatherUsageStats=false'],
        env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            if server.poll() is not None: raise AssertionError((tmp_path/'streamlit.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1): break
            except OSError: time.sleep(.1)
        else: raise AssertionError('Local Streamlit fixture did not start')
        browser_script = r'''
import json,sqlite3,sys
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
from src.runtime.service import SessionClient
port,database,account_id,browser=sys.argv[1:]

def state():
    with sqlite3.connect(database) as db:
        reservations=db.execute('SELECT * FROM reservations ORDER BY intent_id').fetchall()
    return SessionClient(database,account_id=account_id,environment='paper').account(),reservations
baseline=state()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=browser,headless=True)
    page=browser.new_page()
    page.goto('http://127.0.0.1:'+port,wait_until='networkidle')
    actual=page.get_by_role('tab',name='Account suitability',exact=True)
    hypothetical=page.get_by_role('tab',name='Hypothetical capital',exact=True)
    actual.click()
    expect(actual).to_have_attribute('aria-selected','true')
    page.get_by_text('Actual account',exact=True).wait_for()
    assert state()==baseline
    hypothetical.click()
    expect(hypothetical).to_have_attribute('aria-selected','true')
    page.get_by_text('HYPOTHETICAL — execution disabled. These balances are not the actual account.',exact=True).wait_for()
    panel=page.locator('[role="tabpanel"]:visible')
    expect(panel).to_contain_text('SUBSCRIBED_REALTIME')
    expect(panel).to_contain_text('execution_eligible')
    expect(panel).to_contain_text('t59-scenario-5000')
    assert state()==baseline
    actual.click()
    expect(actual).to_have_attribute('aria-selected','true')
    panel=page.locator('[role="tabpanel"]:visible')
    expect(panel).to_contain_text(account_id)
    expect(panel).to_contain_text('10000')
    assert state()==baseline
    expect(page.get_by_text('Data profile: FREE_DELAYED',exact=True)).to_be_visible()
    expect(page.get_by_text('Oldest market as-of',exact=True)).to_be_visible()
    browser.close()
print('T59 actual -> hypothetical -> actual browser switching preserved execution state')
'''
        browser_env=dict(env)
        browser_env['PYTHONPATH']=str(browser_deps)+':'+env['PYTHONPATH']
        completed=subprocess.run([sys.executable,'-c',browser_script,str(port),str(database),account_id,str(browsers[0])],
                                 env=browser_env,capture_output=True,text=True,timeout=60)
        assert completed.returncode==0,completed.stdout+completed.stderr
        assert execution_state()==baseline
    finally:
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired: server.kill();server.wait(timeout=5)
        log.close()


def test_t59_client_scenario_view_reads_preserve_actual_account_and_reservations(tmp_path):
    """CI counterpart: real client comparison and alternating view reads, without a browser."""
    now=datetime(2026,9,9,15,tzinfo=timezone.utc)
    account_id='t59-ci-'+uuid4().hex
    database=tmp_path/'client.sqlite3'
    config=RuntimeConfig(database_path=str(database),account_id=account_id)
    account=AccountProfile(account_id=account_id,environment='paper',hypothetical=False,
        equity='10000',cash='8000',buying_power='8000',currency='USD',reserved_capital='250',
        observed_at=now,mandate_id=config.policy.id)
    scope='paper:'+account_id
    store=Store(database)
    store.project('configuration',scope,config.model_dump(mode='json'),scope)
    store.project('account',scope,account.model_dump(mode='json'),scope)
    store.project('runtime',scope,{'scope':scope,'account_id':account_id,'environment':'paper',
        'execution_mode':'shadow','data_profile':'FREE_DELAYED'},scope)
    with store.transaction():
        store.db.execute('INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?)',
                         ('client-reservation',scope,'session','250','5','TEST','RESERVED',0))
    store.close()
    client=SessionClient(database,account_id=account_id,environment='paper')
    def execution_state():
        with sqlite3.connect(database) as db:
            reservations=db.execute('SELECT * FROM reservations ORDER BY intent_id').fetchall()
        return client.account(),reservations
    before=execution_state()
    scenarios=[AccountProfile.model_validate({**account.model_dump(),'id':uuid4().hex,
        'account_id':'scenario-'+capital,'environment':'scenario','hypothetical':True,
        'equity':capital,'cash':capital,'buying_power':capital,'reserved_capital':'0'}).model_dump(mode='json')
        for capital in ['5000','250000']]
    comparison=client.scenarios(scenarios,data_profile='SUBSCRIBED_REALTIME')
    assert comparison['execution_eligible'] is False
    assert all(row['hypothetical'] for row in comparison['reports'])
    assert execution_state()==before
    for view in ['actual','hypothetical','actual']:
        if view=='actual':
            displayed=client.account()
            assert displayed['account_id']==account_id
            assert displayed['hypothetical'] is False
            assert displayed['equity']=='10000'
        else:
            displayed=client.report()['capital_scenarios'][0]
            assert displayed['execution_eligible'] is False
            assert displayed['data_profile']=='SUBSCRIBED_REALTIME'
        assert execution_state()==before
    assert client.status()['data_profile']=='FREE_DELAYED'
