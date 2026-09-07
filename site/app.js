'use strict';

const menu = document.querySelector('.menu-button');
const navigation = document.querySelector('#navigation');
menu.addEventListener('click', () => {
  const expanded = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(expanded));
  navigation.classList.toggle('open', expanded);
});
navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
  navigation.classList.remove('open');
  menu.setAttribute('aria-expanded', 'false');
}));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
    navigation.classList.remove('open'); menu.setAttribute('aria-expanded', 'false'); menu.focus();
  }
});

function selectButton(selector, selected) {
  document.querySelectorAll(selector).forEach(button => button.setAttribute('aria-pressed', String(button === selected)));
}

const previews = {
  dashboard: ['assets/dashboard-preview.png', 'Gauss World Trader dashboard running the offline session example', 'The real Streamlit dashboard, rendered with a synthetic session fixture. Browse all eight sections in the connected workspace.'],
  terminal: ['assets/terminal-preview.png', 'Gauss World Trader terminal showing a synthetic session heartbeat', 'The real session formatter, rendered with synthetic status records. Health, agent states, entry blockers, and schedule in one view.'],
};
document.querySelectorAll('[data-preview]').forEach(button => button.addEventListener('click', () => {
  selectButton('[data-preview]', button);
  const [src, alt, caption] = previews[button.dataset.preview];
  Object.assign(document.querySelector('#preview-image'), {src, alt});
  document.querySelector('#preview-caption').textContent = caption;
}));

const roles = {
  post: ['EVIDENCE → REVIEW', 'Learn from the completed session.', 'PostGauss reviews completed-session evidence and screens candidates for the next research cycle.', 'Completed-session evidence', 'Reviewed evidence and screened candidates', 'Research only'],
  close: ['REVIEW → CONDITIONAL PLAN', 'Turn research into a testable plan.', 'CloseGauss uses frozen evidence to research conditional plans, with explicit entry conditions and constraints.', 'Frozen research evidence', 'Conditional next-session plans', 'Research only'],
  pre: ['PLAN → READINESS', 'Check the conditions before the open.', 'PreGauss validates plans against funds, permissions, market data, and event readiness. Unmet requirements remain visible.', 'Plans and current account facts', 'Readiness decisions and binding limits', 'Validation gates'],
  live: ['READINESS → SUPERVISION', 'Keep decisions connected to the account.', 'LiveGauss evaluates entry triggers and supervises positions during the session, subject to strategy approvals, data gates, and account limits.', 'Validated plans and market observations', 'Recorded decisions and supervised exposure', 'Shadow by default; approved execution is gated'],
};
document.querySelectorAll('[data-role]').forEach(button => button.addEventListener('click', () => {
  selectButton('[data-role]', button);
  const fields = ['phase', 'title', 'description', 'input', 'output', 'execution'];
  fields.forEach((field, index) => { document.querySelector('#role-' + field).textContent = roles[button.dataset.role][index]; });
}));

let filter = 'all';
const search = document.querySelector('#strategy-search');
function renderStrategies() {
  const query = search.value.trim().toLowerCase();
  const rows = window.GAUSS_STRATEGIES.filter(item => (filter === 'all' || item.asset_type === filter) && `${item.name} ${item.label} ${item.description}`.toLowerCase().includes(query));
  const list = document.querySelector('#strategy-list');
  list.replaceChildren();
  rows.forEach(item => {
    const link = document.createElement('a'); link.className = 'strategy-row';
    link.href = `https://github.com/Magica-Chen/GaussWorldTrader/blob/master/${item.source}`;
    const title = document.createElement('div');
    const heading = document.createElement('h3'); heading.textContent = item.label;
    const code = document.createElement('code'); code.textContent = item.name;
    title.append(heading, code);
    const description = document.createElement('p'); description.textContent = item.description;
    const asset = document.createElement('span'); asset.className = 'asset-tag'; asset.textContent = item.asset_type;
    const arrow = document.createElement('span'); arrow.textContent = '↗'; arrow.setAttribute('aria-hidden', 'true');
    link.append(title, description, asset, arrow); list.append(link);
  });
  document.querySelector('#results-count').textContent = rows.length ? (filter === 'all' && !query ? 'Included strategy examples · View source to explore' : `${rows.length} matching ${rows.length === 1 ? 'example' : 'examples'} · View source to explore`) : 'No matching strategies. Try another search or asset filter.';
}
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {
  filter = button.dataset.filter; selectButton('[data-filter]', button); renderStrategies();
}));
search.addEventListener('input', renderStrategies);
renderStrategies();

const launches = {
  preview: ['python -m streamlit run examples/dashboard_preview.py', 'Explore the real dashboard with synthetic session records. No API keys or broker connection.'],
  dashboard: ['python dashboard.py', 'Open the workspace at localhost:3721. Gauss Session reads the separate service ledger; connected sections use your configured credentials.'],
  research: ['python gauss_bot.py --config examples/gauss.free-delayed.example.toml --once', 'Screen completed daily data, write a next-session report, then exit. Requires provider credentials; submits no orders.'],
  session: ['python gauss_bot.py --config examples/gauss.free-delayed.example.toml', 'Run the persistent four-role service. The example defaults to shadow execution on a paper account and requires provider credentials.'],
  live: ['python live_script.py', 'Review the account and configuration before stock or crypto execution. Options in this menu run underlying research only.'],
  cli: ['python main_cli.py list-strategies', 'Explore the included strategy examples and their interfaces as a starting point for your own implementation. No broker credentials required.'],
};
document.querySelector('#launch-select').addEventListener('change', event => {
  const [command, description] = launches[event.target.value];
  document.querySelector('#launch-code').textContent = command;
  document.querySelector('#launch-description').textContent = description;
});
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const target = document.getElementById(button.dataset.copy);
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(target.textContent);
    status.textContent = 'Copied to clipboard.';
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = button.dataset.copy === 'install-code' ? 'Copy commands' : 'Copy command'; }, 1800);
  } catch {
    const range = document.createRange(); range.selectNodeContents(target);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Clipboard access is unavailable. The command is selected; press Ctrl+C or ⌘C to copy.';
  }
}));
