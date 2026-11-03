"""Optional browser QA against a disposable app root (requires Playwright).

Start a fixture server first; this script uploads synthetic text and starts jobs.
Never run it against a project with unrelated active work.
"""
import argparse
import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright


def main(url, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_selector('.category input')
        assert page.title() == 'Modality · Local preprocessing'
        page.select_option('#upload-category', 'text')
        page.set_input_files('#file-picker', {"name": "browser-test.txt", "mimeType": "text/plain",
                                             "buffer": b'Synthetic browser upload verification.\n'})
        page.wait_for_function("document.getElementById('notice').textContent.includes('Uploaded 1 files')")
        with tempfile.TemporaryDirectory(prefix='folder-upload-') as directory:
            folder = Path(directory)
            (folder/'nested').mkdir()
            (folder/'first.txt').write_text('Synthetic folder fixture one.')
            (folder/'nested/second.md').write_text('Synthetic nested folder fixture two.')
            page.set_input_files('#folder-picker', str(folder))
            page.wait_for_function("document.getElementById('notice').textContent.includes('Uploaded 2 files')")
        assert page.locator('#inputs').inner_text().find('nested/second.md') >= 0
        for checkbox in page.locator('.category input').all():
            checkbox.set_checked(checkbox.get_attribute('value') == 'text')
        page.click('#start')
        page.wait_for_function("document.getElementById('job-status').textContent === 'Completed'", timeout=30000)
        page.locator('#outputs .file-row', has_text='browser-test.txt.txt').get_by_role('button', name='Preview').click()
        page.wait_for_function("document.getElementById('preview-text').textContent.includes('Synthetic browser upload verification')")
        with page.expect_download() as download:
            page.click('#zip')
        assert download.value.suggested_filename.endswith('.zip')
        page.screenshot(path=str(destination/'desktop.png'), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile horizontal overflow'
        page.screenshot(path=str(destination/'mobile.png'), full_page=True)
        browser.close()
    assert not errors, errors
    (destination/'browser-check.json').write_text(json.dumps({"passed": True, "console_errors": errors,
        "checks": ["page load", "file upload", "nested folder upload", "start and poll job", "text preview", "ZIP download", "mobile width"]}, indent=2))
    print('Browser smoke checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', default='runtime/browser-qa')
    args = parser.parse_args()
    main(args.url, args.output)
