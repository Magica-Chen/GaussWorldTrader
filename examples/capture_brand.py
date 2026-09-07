"""Capture the actual offline dashboard and terminal, and check the static site.

Requires Playwright and Chromium: pip install playwright; playwright install chromium.
Run from the repository root: python examples/capture_brand.py
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from io import StringIO
from pathlib import Path

# Repository bootstrap makes these scripts runnable before an editable install.
# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import expect, sync_playwright

from examples.dashboard_preview import PreviewSession
from src.runtime.console import ConsoleOutput
from src.utils.branding import make_console


def capture(review_directory: Path | None = None):
    pictures = ROOT / "docs/images"
    if review_directory is not None:
        review_directory.mkdir(parents=True, exist_ok=True)
    with closing(socket.socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (
        tempfile.TemporaryDirectory(prefix="gauss-brand-") as temp,
        open(Path(temp) / "streamlit.log", "w+") as log,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "examples/dashboard_preview.py",
                "--server.headless=true",
                f"--server.port={port}",
                "--server.address=127.0.0.1",
                "--browser.gatherUsageStats=false",
                "--theme.base=light",
                "--theme.primaryColor=#237f80",
                "--theme.backgroundColor=#f7f9fa",
                "--theme.secondaryBackgroundColor=#ffffff",
                "--theme.textColor=#183448",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Preview server did not start")
            with sync_playwright() as playwright:
                options = {"headless": True}
                if os.environ.get("GAUSS_CHROMIUM"):
                    options["executable_path"] = os.environ["GAUSS_CHROMIUM"]
                browser = playwright.chromium.launch(**options)
                context = browser.new_context(
                    viewport={"width": 1440, "height": 1050}, device_scale_factor=1
                )
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}")
                expect(page.get_by_role("tab", name="Timeline & health")).to_be_visible(
                    timeout=30000
                )
                expect(
                    page.get_by_text(
                        "OFFLINE PREVIEW · SYNTHETIC RECORDS · NO BROKER CONNECTED", exact=True
                    )
                ).to_be_visible()
                assert page.locator('[data-testid="stException"]').count() == 0
                expect(page.get_by_text("python dashboard.py", exact=True)).to_be_visible()
                page.evaluate("document.fonts.ready")
                page.screenshot(path=str(pictures / "dashboard-preview.png"))
                for tab in [
                    "Research & plans",
                    "Account suitability",
                    "Hypothetical capital",
                    "Operator controls",
                    "Timeline & health",
                ]:
                    page.get_by_role("tab", name=tab, exact=True).click()
                    assert page.locator('[data-testid="stException"]').count() == 0
                page.set_viewport_size({"width": 390, "height": 844})
                if review_directory is not None:
                    page.screenshot(path=str(review_directory / "dashboard-mobile.png"))
                page.set_viewport_size({"width": 1440, "height": 1050})

                output = ConsoleOutput("text", StringIO())
                output.console = make_console(
                    file=StringIO(), record=True, force_terminal=True, width=110
                )
                from src.utils.branding import banner

                banner(
                    output.console,
                    "Session monitor",
                    "OFFLINE FIXTURE · SYNTHETIC RECORDS · NO BROKER CONNECTED",
                )
                output(
                    {
                        "event": "heartbeat",
                        "at": "2026-09-04T20:00:00+00:00",
                        "status": PreviewSession().status(),
                    }
                )
                svg = output.console.export_svg(title="Gauss World Trader / Session monitor")
                svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
                if review_directory is not None:
                    (review_directory / "terminal-preview.svg").write_text(svg)
                page.set_content(
                    '<html><body style="margin:0;background:#f7f9fa;display:grid;place-items:center;height:100vh">'
                    + svg
                    + "</body></html>"
                )
                page.screenshot(path=str(pictures / "terminal-preview.png"))

                import shutil

                for name in ["dashboard-preview.png", "terminal-preview.png"]:
                    shutil.copyfile(pictures / name, ROOT / "site/assets" / name)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto((ROOT / "site/index.html").as_uri())
                expect(page.locator(".strategy-row")).to_have_count(12)
                for asset, count in [("stock", 8), ("crypto", 2), ("option", 2), ("all", 12)]:
                    page.locator(f'[data-filter="{asset}"]').click()
                    expect(page.locator(".strategy-row")).to_have_count(count)
                page.locator("#strategy-search").fill("momentum")
                expect(page.locator(".strategy-row")).to_have_count(2)
                page.locator("#strategy-search").fill("not-a-strategy")
                expect(page.locator("#results-count")).to_contain_text("No matching strategies")
                page.locator("#strategy-search").fill("")
                for role in ["live", "pre", "close", "post"]:
                    page.locator(f'[data-role="{role}"]').click()
                    expect(page.locator(f'[data-role="{role}"]')).to_have_attribute(
                        "aria-pressed", "true"
                    )
                page.locator('[data-preview="terminal"]').click()
                expect(page.locator("#preview-image")).to_have_attribute(
                    "src", "assets/terminal-preview.png"
                )
                page.locator('[data-preview="dashboard"]').click()
                for launch in ["dashboard", "research", "session", "live", "cli", "preview"]:
                    page.locator("#launch-select").select_option(launch)
                    expect(page.locator("#launch-code")).not_to_be_empty()
                page.locator('[data-copy="launch-code"]').click()
                expect(page.locator("#copy-status")).not_to_be_empty()
                assert not errors, errors
                assert page.locator("img").evaluate_all(
                    "(images) => images.every(img => img.complete && img.naturalWidth > 0)"
                )
                page.evaluate(
                    "window.getSelection().removeAllRanges(); document.activeElement.blur(); document.querySelector('#copy-status').textContent = ''; window.scrollTo({top:0,behavior:'instant'})"
                )
                page.mouse.move(0, 0)
                if review_directory is not None:
                    page.screenshot(path=str(review_directory / "site-desktop.png"), full_page=True)
                for width in [390, 768, 1024, 1440]:
                    page.set_viewport_size({"width": width, "height": 844})
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    ), f"Horizontal overflow at {width}px"
                page.set_viewport_size({"width": 390, "height": 844})
                page.locator(".menu-button").click()
                expect(page.locator("#navigation")).to_be_visible()
                page.keyboard.press("Escape")
                expect(page.locator(".menu-button")).to_have_attribute("aria-expanded", "false")
                page.locator(".menu-button").blur()
                page.evaluate("window.scrollTo({top:0,behavior:'instant'})")
                if review_directory is not None:
                    page.screenshot(path=str(review_directory / "site-mobile.png"), full_page=True)
                browser.close()
            print(
                "Captured real dashboard and terminal. Site filters, roles, previews, commands, clipboard, assets and responsive layouts passed."
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir", type=Path, help="Optional output directory for review-only captures"
    )
    capture(parser.parse_args().review_dir)
