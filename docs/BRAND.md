# Gauss World Trader visual identity

The identity retains the project name and follows `assets/temp/logo-new3.png`: a
Gaussian curve, ascending bars, deep navy, and muted teal. Reference logos remain
untouched. The new mark and banner were generated with the built-in imagegen tool.

Present the included basic strategies as examples and starting points. Emphasize
the ability to build and integrate custom strategies; avoid strategy totals as a
headline, badge, or product selling point.

## Shared design

| Element | Choice |
|---|---|
| Navy | `#183448` — primary text and terminal surfaces |
| Teal | `#237f80` — emphasis, active states, and charts |
| Paper | `#f7f9fa` — light workspace background |
| Secondary text | `#607480` |
| Borders | `#dce5e8` |
| Typography | System sans serif; monospace for commands and identifiers |
| Layout | Generous spacing, fine rules, limited decoration, visible focus states |

The Streamlit theme lives in `src/ui/brand.py` and `src/ui/assets/brand.css`.
The launcher sets the light palette. Shared terminal presentation lives in
`src/utils/branding.py`; Rich respects terminal color capabilities and `NO_COLOR`.
Continuous redirected bot output remains JSON, and `--output json` remains available.

## Assets

| File | Purpose |
|---|---|
| `assets/brand/gauss-mark.png` | Transparent Gaussian brand symbol |
| `assets/brand/gauss-banner.png` | README banner and website social image |
| `src/ui/assets/gauss-mark.png` | Packaged dashboard logo and page icon |
| `docs/images/dashboard-preview.png` | Production Streamlit session view with synthetic records |
| `docs/images/terminal-preview.png` | Production console formatter with synthetic records |
| `docs/images/terminal-preview.svg` | Scalable source for the terminal capture |
| `docs/images/site-desktop.png`, `site-mobile.png` | Website review captures |
| `docs/images/dashboard-mobile.png` | Mobile dashboard review capture |

Website copies live in `site/assets/`. Run `python examples/refresh_brand.py` after
updating the canonical artwork or captures. The command also refreshes the strategy
library from the actual registry and validates its source paths. Dashboard assets
are included in the Python wheel through `pyproject.toml` package-data rules.

## Preview and capture

After installing the runtime requirements:

```bash
# Production session UI, synthetic records, no provider credentials needed.
python -m streamlit run examples/dashboard_preview.py

# The full connected workspace.
python dashboard.py

# Serve GitHub Pages locally.
python -m http.server 8080 --directory site
```

The offline example never creates provider or broker clients. Operator commands are
disabled. It shows the session section; the connected dashboard retains all eight
sections. Images are application captures, not rendered mockups. Sample balances,
market times, and plans are explicitly synthetic.

To reproduce the screenshots and browser interaction checks:

```bash
python -m pip install playwright
python -m playwright install chromium
python examples/capture_brand.py
python examples/refresh_brand.py
```

`GAUSS_CHROMIUM` can point to an existing Chromium executable. The capture script
starts and stops its own local preview server. It checks session tabs, all strategy
filters, search and empty results, each workflow role, both interface previews,
launch choices, clipboard feedback, local image loading, mobile navigation, and
page overflow at 390, 768, 1024, and 1440 pixels. Clipboard-denied browsers receive
selected command text and keyboard-copy instructions.

The website uses plain HTML, CSS, and JavaScript and has no build step or remote
font dependency. Keyboard users can reach every interactive control; the mobile
menu supports Escape. Reduced-motion preferences disable smooth scrolling.
The existing `.github/workflows/pages.yml` publishes `site/` on qualifying pushes
to `master` or manual workflow dispatch. Local edits do not deploy the public site.

## Generation prompts

Both assets used the built-in imagegen tool. The PNG outputs were copied into the
repository unchanged, preserving the logo's alpha channel.

### Gaussian mark

Reference: `assets/temp/logo-new3.png`.

> Create a clean professional brand illustration for Gauss World Trader, a quantitative research and trading toolkit. Use case: logo-brand. Create ONE isolated Gaussian bell curve symbol in deep navy #142F43, enclosing four rectangular bars of ascending height from left to right in muted teal #237F80. Smooth symmetric bell curve and geometric bars, flat vector-like raster design, no gradient, no texture, no shadow, no text or letters. Large centered symbol with minimal 10 percent padding, square transparent canvas with actual alpha. The supplied image is a motif reference; omit its wordmark.

### README banner

> Use case: ads-marketing. Create a refined horizontal 3:1 brand banner for the README of Gauss World Trader, a professional open-source quantitative trading toolkit. Off-white #f7f9fa background, generous white space. Left half: sophisticated deep navy typography, exact title 'Gauss World Trader' with 'Gauss' bold, smaller subtitle 'Research. Validate. Execute.' Right half: an elegant translucent teal Gaussian bell-shaped mathematical surface on a subtle fine grey analytical grid, restrained dimensional illustration, no stock price predictions, no digits, no arrows, no coins. Palette navy #183448, muted teal #237f80, soft silver grey. Editorial design, quiet and precise, no clutter, no glow, no heavy drop shadows. The composition must feel like a serious quantitative research brand.

## Verification

Verified locally on Python 3.13:

- The repository's explicit offline suite: **237 passed**.
- All eight production dashboard navigation routes dispatch with connected providers mocked.
- Browser checks cover desktop/mobile layouts and every website control described above.
- The built wheel imports outside the checkout and includes the logo and theme files.
- Backtest tables, live configuration, and one-shot reports render with synthetic inputs.
- Continuous JSON output and the one-shot JSON path pass the existing regression checks.
- Local documentation links, copied assets, generated PNG alpha, and JavaScript syntax validate.

No connected trading session was started. The public GitHub Pages deployment remains
controlled by the repository's existing workflow.
