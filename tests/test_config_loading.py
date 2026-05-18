import tempfile
import unittest
from pathlib import Path

from rfo_demo.utils.config import load_python_config


class ConfigLoadingTest(unittest.TestCase):
    def test_load_config_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.py"
            path.write_text("CONFIG = {'a': 1, 'b': 'x'}\n", encoding="utf-8")
            config = load_python_config(str(path))
        self.assertEqual(config["a"], 1)
        self.assertEqual(config["b"], "x")


if __name__ == "__main__":
    unittest.main()
