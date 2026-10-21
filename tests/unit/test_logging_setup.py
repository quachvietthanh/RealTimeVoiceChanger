"""Unit test for the backend log file setup."""

import logging
import sys
import tempfile
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.diagnostics.logging_setup import configure_logging


class LoggingSetupTests(unittest.TestCase):
    def test_writes_backend_errors_to_file_once(self):
        with tempfile.TemporaryDirectory() as d:
            logger = logging.getLogger("voice_changer")
            before = list(logger.handlers)
            try:
                path = configure_logging(Path(d))
                configure_logging(Path(d))  # idempotent
                added = [h for h in logger.handlers if h not in before]
                self.assertEqual(len(added), 1)
                logging.getLogger("voice_changer.service.service").error("model load failed for profile x")
                added[0].flush()
                self.assertIn("model load failed for profile x", path.read_text(encoding="utf-8"))
            finally:
                for h in logger.handlers[:]:
                    if h not in before:
                        logger.removeHandler(h)
                        h.close()


if __name__ == "__main__":
    unittest.main()
