import os, tempfile, unittest
from pathlib import Path
from spaceslug.rendered_page import SubprocessPageRenderer

class RenderTest(unittest.TestCase):
    def test_missing_is_explicit(self):
        r=SubprocessPageRenderer('/missing').render('https://example.org/docs')
        self.assertEqual(r.status,'unavailable')
    def test_isolated_fake_renderer(self):
        with tempfile.TemporaryDirectory() as d:
            script=Path(d)/'renderer'; script.write_text('#!/bin/sh\nwhile [ "$1" != "--output" ]; do shift; done; shift; printf rendered > "$1"\n'); script.chmod(0o700)
            r=SubprocessPageRenderer(script,allowed_origins=('https://example.org',)).render('https://example.org/docs')
            self.assertEqual((r.status,r.text),('ok','rendered'))
            self.assertEqual(SubprocessPageRenderer(script,allowed_origins=('https://other.org',)).render('https://example.org/docs').status,'rejected')

if __name__=='__main__': unittest.main()
