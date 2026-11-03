"""Offline behavior checks: no inference, downloads, or running Ollama needed."""
import io
import json
from pathlib import Path
import tempfile
import shutil
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
from fastapi.testclient import TestClient
from PIL import Image
from app.config import Settings
from app.engines import reading_positions, normalize_boxes
from app.files import safe_path, list_files, save_json
from app.jobs import Jobs
from app.main import create_app
from app.processors import Processor
from app.processors import command
from app.results import Result


class FakeEngines:
    def __init__(self):
        self.events = []
        self.calls = []
        self.fail_vision = False

    def unload_vision(self):
        pass

    def ocr(self, path):
        self.calls.append('ocr')
        return [{"text": "second", "confidence": .9, "bbox": [5, 30, 80, 50]},
                {"text": "first", "confidence": .9, "bbox": [5, 5, 80, 20]}]

    def layout(self, path):
        self.calls.append('layout')
        return [{"label": "chart", "confidence": .9, "bbox": [0, 0, 100, 100]}]

    def order(self, regions, width, height):
        self.calls.append('order')
        return [1, 0]

    def describe(self, image, text, kind):
        self.calls.append(('vision', text))
        if self.fail_vision:
            raise RuntimeError('vision unavailable')
        return 'A visible chart.', False

    def transcribe(self, path):
        self.calls.append('transcribe')
        return [{"start": 0, "end": 1, "text": "Synthetic test transcript"}], 1.0


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(root=Path(self.temp.name))
        self.settings.initialize()
        self.engine = FakeEngines()
        self.processor = Processor(self.settings, self.engine)

    def test_unsafe_paths_and_symlinks(self):
        root = self.settings.root / 'output'
        for path in ['../secret', '/etc/passwd', 'pdf/../../secret', 'C:\\file', 'pdf/.secret', 'pdf/./x']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_path(root, path)
        with tempfile.TemporaryDirectory() as outside:
            (root / 'pdf/link').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, 'pdf/link/a.txt')

    def test_text_is_not_rewritten(self):
        source = self.settings.root / 'input/text/note.md'
        source.write_text('# Original\n  punctuation, and spaces!\n', encoding='utf-8')
        result = self.processor.process(source, 'text')
        self.assertEqual(result.sections, [source.read_text()])
        self.assertFalse(result.warnings)
        self.assertEqual(self.engine.calls, [])

    def test_csv_preserves_quoted_cells_and_rows(self):
        source = self.settings.root / 'input/excel/table.csv'
        source.write_text('name,note\nAlice,"a,b"\n', encoding='utf-8')
        result = self.processor.process(source, 'excel')
        self.assertIn('["Alice", "a,b"]', result.sections[1])

    def test_xlsx_formulas_kept_when_visual_conversion_fails(self):
        from openpyxl import Workbook
        source = self.settings.root / 'input/excel/book.xlsx'
        workbook = Workbook(); workbook.active.title = 'Budget'
        workbook.active.append(['Amount', 3, '=B1*2']); workbook.save(source); workbook.close()
        with patch.object(self.processor, 'office_pdf', side_effect=RuntimeError('conversion unavailable')):
            result = self.processor.process(source, 'excel')
        self.assertIn('Budget', '\n'.join(result.sections))
        self.assertIn('=B1*2 [cached value: unavailable]', '\n'.join(result.sections))
        self.assertTrue(result.warnings)

    def test_ocr_layout_reader_vision_sequence_and_prompt(self):
        source = self.settings.root / 'input/image/figure.png'
        Image.new('RGB', (100, 100), 'white').save(source)
        result = self.processor.process(source, 'image')
        self.assertEqual(self.engine.calls[:3], ['ocr', 'layout', 'order'])
        self.assertEqual(self.engine.calls[3], ('vision', 'first\nsecond'))
        self.assertIn('first\nsecond', result.sections[0])
        self.assertIn('VLM description (unverified)', result.sections[1])

    def test_no_ocr_still_calls_vision(self):
        self.engine.ocr = lambda path: []
        source = self.settings.root / 'input/image/photo.png'
        Image.new('RGB', (100, 100)).save(source)
        result = self.processor.process(source, 'image')
        self.assertIn('No text recognized', result.sections[0])
        self.assertTrue(any(isinstance(c, tuple) for c in self.engine.calls))

    def test_partial_vision_failure_retains_ocr(self):
        self.engine.fail_vision = True
        source = self.settings.root / 'input/image/report.png'
        Image.new('RGB', (100, 100)).save(source)
        result = self.processor.process(source, 'image')
        self.assertIn('first', result.text(source.name))
        self.assertIn('PARTIAL', result.text(source.name))
        self.assertIn('vision unavailable', result.warnings[0])

    def test_reader_limit_does_not_drop_ocr(self):
        self.engine.order = lambda *args: (_ for _ in ()).throw(ValueError('510-region limit'))
        source = self.settings.root / 'input/image/report.png'
        Image.new('RGB', (100, 100)).save(source)
        result = self.processor.process(source, 'image')
        self.assertIn('second\nfirst', result.sections[0])
        self.assertIn('510-region limit', '\n'.join(result.warnings))

    def test_all_tiff_pages_are_processed(self):
        source = self.settings.root / 'input/image/pages.tiff'
        a = Image.new('RGB', (100, 100)); b = Image.new('RGB', (100, 100))
        a.save(source, save_all=True, append_images=[b])
        result = self.processor.process(source, 'image')
        self.assertEqual(len(result.details), 2)
        self.assertEqual([d['reference'] for d in result.details], ['Page 1', 'Page 2'])

    def test_pdf_native_text_and_page_references(self):
        import pymupdf
        source = self.settings.root / 'input/pdf/two.pdf'
        with pymupdf.open() as doc:
            for text in ['Native first page', 'Native second page']:
                page = doc.new_page(); page.insert_text((30, 50), text)
            doc.save(source)
        result = self.processor.process(source, 'pdf')
        self.assertIn('Native first page', result.text(source.name))
        self.assertIn('Page 2', result.text(source.name))
        self.assertEqual(len(result.details), 2)

    def test_reading_assignment_is_a_permutation(self):
        import numpy as np
        positions = reading_positions(np.array([[5, 2, 1], [4, 3, 2], [8, 1, 4]]))
        self.assertEqual(sorted(positions), [0, 1, 2])
        self.assertEqual(normalize_boxes([[10, 20, 100, 80]], 200, 100), [[50, 200, 500, 800]])

    def test_unsupported_and_empty_sources(self):
        with self.assertRaises(ValueError):
            self.processor.process(Path('bad.exe'), 'image')
        source = self.settings.root / 'input/text/empty.txt'; source.touch()
        self.assertTrue(self.processor.process(source, 'text').warnings)

    @unittest.skipUnless(shutil.which('libreoffice'), 'LibreOffice integration needs the Docker image')
    def test_xlsx_chart_rendering_and_native_cells(self):
        from openpyxl import Workbook
        from openpyxl.chart import BarChart, Reference
        source = self.settings.root / 'input/excel/chart.xlsx'
        workbook = Workbook(); sheet = workbook.active; sheet.title = 'Synthetic Sales'
        sheet.append(['Month', 'Amount']); sheet.append(['January', 12]); sheet.append(['February', 20])
        chart = BarChart(); chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
        sheet.add_chart(chart, 'D2'); workbook.save(source); workbook.close()
        result = self.processor.process(source, 'excel')
        self.assertFalse(result.warnings, result.warnings)
        self.assertIn('Synthetic Sales', result.text(source.name))
        self.assertIn('VLM description (unverified)', result.text(source.name))

    @unittest.skipUnless(shutil.which('libreoffice'), 'LibreOffice integration needs the Docker image')
    def test_docx_converts_to_document_pipeline(self):
        source = self.settings.root / 'input/docs/test.docx'
        with zipfile.ZipFile(source, 'w') as doc:
            doc.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
            doc.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
            doc.writestr('word/document.xml', '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Synthetic Word fixture</w:t></w:r></w:p></w:body></w:document>')
        result = self.processor.process(source, 'docs')
        self.assertFalse(result.warnings, result.warnings)
        self.assertIn('Synthetic Word fixture', result.text(source.name))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Media integration needs the Docker image')
    def test_video_audio_and_sampled_visuals(self):
        source = self.settings.root / 'input/video/test.mp4'
        command(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=160x120:r=5',
                 '-f','lavfi','-i','sine=frequency=440:sample_rate=16000','-t','2.2','-c:v','mpeg4','-c:a','aac',source])
        self.settings.frame_interval = 2
        result = self.processor.process(source, 'video')
        self.assertFalse(result.warnings, result.warnings)
        self.assertIn('Synthetic test transcript', result.text(source.name))
        self.assertEqual(len([d for d in result.details if 'reference' in d]), 2)
        self.assertIn('00:00:02.000', result.text(source.name))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Media integration needs the Docker image')
    def test_silent_video_still_analyzes_frames(self):
        source = self.settings.root / 'input/video/silent.mp4'
        command(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=blue:s=160x120:r=5',
                 '-t','1','-c:v','mpeg4',source])
        result = self.processor.process(source, 'video')
        self.assertFalse(result.warnings, result.warnings)
        self.assertIn('no audio track', result.text(source.name))
        self.assertIn('VLM description', result.text(source.name))


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.settings = Settings(root=Path(self.temp.name)); self.settings.initialize()
        self.context = TestClient(create_app(self.settings)); self.client = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def wait(self, job_id):
        for _ in range(200):
            report = self.client.get('/api/jobs/' + job_id).json()
            if report['status'] not in ['queued', 'running']:
                return report
            time.sleep(.01)
        self.fail('Job failed to finish')

    def test_home_and_health(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/api/health').json()['status'], 'ok')
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)

    def test_upload_subfolders_process_preview_and_zip(self):
        response = self.client.post('/api/upload/text', files=[('files', ('note.txt', b'Hello source'))],
                                    data={'paths': json.dumps(['folder/note.txt'])})
        self.assertEqual(response.status_code, 200, response.text)
        start = self.client.post('/api/jobs', json={'categories': ['text']})
        self.assertEqual(start.status_code, 202, start.text)
        job = self.wait(start.json()['id'])
        self.assertEqual(job['status'], 'completed')
        path = 'text/folder/note.txt.txt'
        self.assertIn('Hello source', self.client.get('/api/preview', params={'path': path}).text)
        archive = self.client.get('/api/jobs/' + job['id'] + '/download')
        with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
            self.assertIn(path, z.namelist())
            self.assertIn('Hello source', z.read(path).decode())

    def test_upload_rejects_traversal_extension_and_mismatched_paths(self):
        for category, paths in [('text', ['../secret.txt']), ('image', ['fake.exe']), ('text', ['a.txt', 'b.txt'])]:
            response = self.client.post('/api/upload/'+category,
                files=[('files', ('note.txt', b'hello'))], data={'paths': json.dumps(paths)})
            self.assertEqual(response.status_code, 400, response.text)

    def test_upload_size_limit(self):
        self.settings.max_upload_mb = 1
        response = self.client.post('/api/upload/text', files={'files': ('large.txt', b'a'*(1024**2+1))})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list_files(self.settings, 'input'), [])

    def test_download_rejects_outside_paths(self):
        self.assertEqual(self.client.get('/api/download', params={'path': '../secret.txt'}).status_code, 400)
        self.assertEqual(self.client.get('/api/preview', params={'path': 'pdf/missing.txt'}).status_code, 404)

    def test_batch_continues_after_bad_utf8(self):
        root = self.settings.root / 'input/text'
        (root / 'a.txt').write_bytes(b'\xff\xff'); (root / 'b.txt').write_text('Valid')
        job = self.wait(self.client.post('/api/jobs', json={'categories': ['text']}).json()['id'])
        self.assertEqual([f['status'] for f in job['files']], ['failed', 'completed'])
        self.assertEqual(job['status'], 'partial')

    def test_same_stem_different_extensions_and_rerun(self):
        root = self.settings.root / 'input/text'
        (root / 'note.txt').write_text('one'); (root / 'note.md').write_text('two')
        job = self.wait(self.client.post('/api/jobs', json={'categories': ['text']}).json()['id'])
        self.assertEqual(len(list_files(self.settings, 'output')), 2)
        (root / 'note.txt').write_text('three')
        self.wait(self.client.post('/api/jobs', json={'categories': ['text']}).json()['id'])
        self.assertIn('three', (self.settings.root/'output/text/note.txt.txt').read_text())
        archive = self.client.get('/api/jobs/' + job['id'] + '/download')
        with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
            old = z.read('text/note.txt.txt').decode()
            self.assertIn('one', old)
            self.assertNotIn('three', old)

    def test_empty_or_invalid_job_request(self):
        self.assertEqual(self.client.post('/api/jobs', json={'categories': []}).status_code, 400)
        self.assertEqual(self.client.post('/api/jobs', json={'categories': ['invalid']}).status_code, 400)
        self.assertEqual(self.client.post('/api/jobs', json={'categories': ['text']}).status_code, 400)


class JobStateTests(unittest.TestCase):
    def test_single_worker_busy_and_restart_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(root=Path(directory)); settings.initialize()
            (settings.root/'input/text/file.txt').write_text('hello')
            entered, release = threading.Event(), threading.Event()
            class SlowProcessor:
                def __init__(self, *args, **kwargs): pass
                def process(self, *args):
                    entered.set(); release.wait(timeout=5); return Result(sections=['ok'])
            jobs = Jobs(settings, SlowProcessor)
            try:
                first = jobs.start(['text']); self.assertTrue(entered.wait(2))
                with self.assertRaises(RuntimeError): jobs.start(['text'])
            finally:
                release.set(); jobs.close()
            report = jobs.get(first['id']); report['status']='running'; report['files'][0]['status']='running'
            save_json(settings.root/'runtime/jobs'/f"{report['id']}.json", report)
            restored = Jobs(settings)
            try:
                self.assertEqual(restored.get(first['id'])['status'], 'interrupted')
                self.assertEqual(restored.get(first['id'])['files'][0]['status'], 'interrupted')
            finally: restored.close()


if __name__ == '__main__':
    unittest.main()
