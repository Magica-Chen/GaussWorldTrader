# Gauss World Trader visual identity

The identity combines a Gaussian distribution, financial candlesticks, and an upward
trend. The mark develops the earlier logo-new2 reference into a standalone symbol.
The banner retains its left-hand typography and projects a world map onto the
Gaussian surface on the right, with candlesticks and a rising arrow inside the
distribution.

Present the included basic strategies as examples and starting points. Emphasize
the ability to build and integrate custom strategies; avoid strategy totals as a
headline, badge, or product selling point.

## Shared design

| Element | Choice |
|---|---|
| Navy | `#183448` — primary text and the Gaussian curve |
| Teal | `#237f80` — upward trend, emphasis, and charts |
| Slate blue | Candlestick detail in the mark |
| Paper | `#f7f9fa` — light workspace background |
| Secondary text | `#607480` |
| Borders | `#dce5e8` |
| Typography | System sans serif; monospace for commands and identifiers |
| Layout | Generous spacing, fine rules, limited decoration, visible focus states |

The Streamlit theme lives in `src/ui/brand.py` and `src/ui/assets/brand.css`.
The launcher sets the light palette. Shared terminal presentation lives in
`src/utils/branding.py`; Rich respects terminal color capabilities and `NO_COLOR`.
Continuous redirected bot output remains JSON, and `--output json` remains available.

## Active assets

| File | Purpose |
|---|---|
| `assets/brand/gauss-mark.png` | Transparent Gaussian curve, candlesticks, and upward trend symbol |
| `assets/brand/gauss-banner.png` | README banner and website social image |
| `src/ui/assets/gauss-mark.png` | Packaged dashboard logo and page icon |
| `docs/images/dashboard-preview.png` | Production Streamlit session view with synthetic records |
| `docs/images/terminal-preview.png` | Production console formatter with synthetic records |

Website copies live in `site/assets/`. Run `python examples/refresh_brand.py` after
updating the canonical artwork or captures. The command also refreshes the strategy
library from the actual registry and validates its source paths. Dashboard assets
are included in the Python wheel through `pyproject.toml` package-data rules.

Superseded logos, design references, old interface screenshots, and standalone review
captures have been removed from the repository. Keep current assets and the copies
required by the packaged dashboard and static website.

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
sections. Sample balances, market times, and plans are explicitly synthetic.

To reproduce the active screenshots and browser interaction checks:

```bash
python -m pip install playwright
python -m playwright install chromium
python examples/capture_brand.py
python examples/refresh_brand.py
```

For optional desktop/mobile review captures and the terminal SVG, choose a separate
output directory:

```bash
python examples/capture_brand.py --review-dir /tmp/gauss-brand-review
```

`GAUSS_CHROMIUM` can point to an existing Chromium executable. The capture script
starts and stops its own local preview server. It checks session tabs, strategy
filters, search and empty results, workflow roles, interface previews, launch
choices, clipboard feedback, image loading, mobile navigation, and page overflow
at 390, 768, 1024, and 1440 pixels.

The website uses plain HTML, CSS, and JavaScript, with no build step or remote font
dependency. Keyboard users can reach every interactive control; the mobile menu
supports Escape. Reduced-motion preferences disable smooth scrolling.
The existing `.github/workflows/pages.yml` publishes `site/` on qualifying pushes
to `master` or manual workflow dispatch.

## Generation prompts

Both current assets used the built-in imagegen tool. The PNG outputs were copied
into the repository unchanged, preserving the mark's alpha channel. The reference
artwork was used during generation and removed during the requested cleanup.

### Gaussian mark

> Use case: logo-brand.
> Asset type: replacement standalone transparent brand mark for Gauss World Trader, used at small size in a dashboard and website.
> Input image: logo-new2, a reference for the LEFT-HAND symbol only. Its wordmark is not part of this deliverable.
> Primary request: develop that left symbol into one clean professional logo combining THREE clearly legible elements: (1) a deep navy Gaussian probability distribution bell curve, with recognizable symmetric rounded peak and low tails; (2) a small candlestick chart with four or five separated rectangular candles with visible thin upper and lower wicks, not plain histogram bars; and (3) a bold muted teal rising zigzag trend line ending in ONE upward-right arrowhead, crossing through the composition with clean negative-space separation like the reference.
> Style: crisp geometric vector-like raster artwork with solid navy #183448, teal #237f80, and medium slate blue. Restrained, elegant, flat, coherent line weights. Candles should be substantial and legible, not pale or tiny. No text, letters, numbers, globe, framing box, gradients, glow, texture or shadows.
> Composition: centered compact symbol, occupies about 85 percent of the square canvas width and 70 percent of height, with clear space around every tip. Truly transparent background, actual alpha channel. The three motifs must remain recognizable as a single integrated logo.

### Banner revision

> Use case: precise-object-edit.
> Asset type: Gauss World Trader website and README banner, wide 3:1 format.
> Input image: edit target, the ORIGINAL banner with its Gaussian surface. Use this exact composition as the starting point.
>
> Primary request: Keep the original two-variable Gaussian distribution height plot on the right. Project a recognizable flat WORLD MAP onto that same bell-shaped surface, and plot financial candlesticks INSIDE the translucent Gaussian volume. The result is one integrated mathematical chart.
>
> Preserve: all of the left-side typography, positions, spacing, font weights, horizontal rule, pale background, and exact text "Gauss World Trader", "Research. Validate. Execute.", and "OPEN-SOURCE QUANTITATIVE TRADING TOOLKIT". Keep the Gaussian's existing smooth bell silhouette, wide tails on the horizontal plane, mesh, scale, location, and oblique camera view.
>
> World map: drape a two-dimensional world map across the Gaussian surface as a restrained teal geographic texture with clearly recognizable continent coastlines. The Americas occupy the left-facing slope; Europe and Africa the central/front slope; Asia the right-facing slope. The coastlines follow the curvature and perspective of the Gaussian mesh, extending smoothly down its slopes. The map is printed on the existing surface, with subtle translucent continent fills and fine darker outlines. It must read as geography on a Gaussian height plot.
>
> Candlestick chart: show a single sequence of seven clear financial candles INSIDE the bell, visible THROUGH its translucent mapped surface, beneath the outer bell profile. Each candle has a rectangular body and thin upper and lower wicks. Place them on a shared internal vertical plotting plane that follows the base-grid perspective, spanning the lower central/front volume from left to right. Use restrained dark teal and muted slate candles, with varying bodies and wicks and a generally rising progression. Keep all candles within the bell's silhouette. A fine rising arrow may connect the series inside the same plotting volume. Give the candles enough contrast to remain readable through the lightly tinted map and mesh; geographic texture and candles should both be recognizable.
>
> Style: match the original clean, airy mathematical visualization, delicate teal wireframe and translucent surface, navy typography, off-white background, subtle contour lines and ground grid. Keep the right-hand composition compact and uncluttered.
>
> Avoid: any spherical Earth, globe, rounded underside, pear shape, closed glass sculpture, separate map panel, floating globe, separate chart panel, or floating extra Gaussian. Do not reshape the original Gaussian into a planet. Do not add labels, numbers, extra text, icons, neon, or decoration.

## Verification

The initial rebrand passed the repository's 237-test offline suite, all eight
dashboard navigation routes with connected providers mocked, and an installed-wheel
check for packaged assets. The artwork revision is checked through the browser
capture workflow, image inspection, asset synchronization, and local reference
validation. No connected trading session is needed for these checks.
