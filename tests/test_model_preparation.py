import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.postprocess.integrity import sha256
from scripts.prepare_predictions import MODEL_PROFILES, prepare_model


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw"
        self.source.mkdir()

    def test_model_routing_and_logged_implementation(self):
        for model, profile in MODEL_PROFILES.items():
            with self.subTest(model=model):
                source = self.root / model
                source.mkdir()
                if model == "surya-ocr-2":
                    (source / "a.json").write_text(
                        json.dumps(
                            {"blocks": [{"html": '<math display="block">x</math>'}]}
                        )
                    )
                    expected = b"$$\nx\n$$\n"
                else:
                    (source / "a.md").write_bytes(b"$$x$$ (1)\r\nrepeat\r\nrepeat\r\n")
                    expected = (
                        b"$$x$$\n\n(1)\nrepeat\nrepeat\n"
                        if model == "oxr-1.0"
                        else (source / "a.md").read_bytes()
                    )
                output = self.root / (model + "-prepared")
                audit = prepare_model(source, output, model, 1)
                self.assertEqual((output / "a.md").read_bytes(), expected)
                self.assertEqual(audit["model"], model)
                self.assertEqual(audit["profile"], profile)
                implementation = Path(__file__).parents[1] / "scripts"
                self.assertEqual(
                    audit["implementation_sha256"]["prepare_predictions.py"],
                    sha256(implementation / "prepare_predictions.py"),
                )
                self.assertEqual(
                    audit["implementation_sha256"]["prepare.py"],
                    sha256(implementation / "postprocess" / "prepare.py"),
                )
                self.assertFalse(audit["ground_truth_used"])
                self.assertFalse(audit["scoring_modified"])

    def test_rejects_unknown_model_before_writing(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            prepare_model(self.source, self.root / "output", "unknown", 1)
        self.assertFalse((self.root / "output").exists())

    def test_cli_dispatch_and_wrong_schema(self):
        (self.source / "a.md").write_text("$$x$$ (1)\n")
        output = self.root / "output"
        command = [
            sys.executable,
            "-m",
            "scripts.prepare_predictions",
            "--source",
            str(self.source),
            "--output",
            str(output),
            "--expected-count",
            "1",
        ]
        failed = subprocess.run(
            command + ["--model", "surya-ocr-2"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(output.exists())
        passed = subprocess.run(
            command + ["--model", "oxr-1.0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertEqual(json.loads(passed.stdout)["profile"], "display-math-v1")


if __name__ == "__main__":
    unittest.main()
