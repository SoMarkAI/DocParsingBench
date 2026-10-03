import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from reproduction.display_math_v1 import isolate_display_math
from reproduction.integrity import inventory, sha256
from reproduction.prepare import prepare
from reproduction.surya_html_v1 import serialize_page
from reproduction.verify import verify


class DisplayMathTests(unittest.TestCase):
    def test_historical_examples(self):
        cases = [
            (
                "Intro\n\n$$\na+b=c\n$$\n(6)\n\nTail\n",
                "Intro\n\n$$\na+b=c\n$$\n\n(6)\n\nTail\n",
            ),
            ("Intro\n\n$$a+b=c$$\n\nTail\n", "Intro\n\n$$a+b=c$$\n\nTail\n"),
            (
                "<table><tr><td>$$x$$ text</td></tr></table>\n",
                "<table><tr><td>$$x$$ text</td></tr></table>\n",
            ),
            ("$$a$$ (1) $$b$$ (2)\n", "$$a$$\n\n(1)\n\n$$b$$\n\n(2)\n"),
            ("$$a$$ (1) $$b$$\n", "$$a$$\n\n(1)\n\n$$b$$\n"),
            ("a+b=c\\]\n", "a+b=c\\]\n"),
            ("text \\[x+y\\] (3)\n", "text\n\n\\[x+y\\]\n\n(3)\n"),
        ]
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(isolate_display_math(original), expected)
                self.assertEqual(isolate_display_math(expected), expected)

    def test_repetitions_are_not_removed(self):
        source = "repeated\nrepeated\n$$x$$ (1) $$x$$ (1)\n"
        output = isolate_display_math(source)
        self.assertEqual(output.count("repeated"), 2)
        self.assertEqual(output.count("$$x$$"), 2)
        self.assertEqual(output.count("(1)"), 2)


class SuryaTests(unittest.TestCase):
    def test_math_heading_order_and_entities(self):
        text, stats = serialize_page(
            {
                "blocks": [
                    {
                        "reading_order": 2,
                        "html": '<math display="block">a&amp;b</math>',
                    },
                    {
                        "reading_order": 1,
                        "html": "<h2>A&amp;B</h2>",
                        "label": "SectionHeader",
                    },
                    {"reading_order": 3, "html": "<p>x <math>y</math></p>"},
                ]
            }
        )
        self.assertEqual(text, "## A&B\n\n$$\na&b\n$$\n\n<p>x $y$</p>")
        self.assertEqual(stats["display_math_tags_converted"], 1)
        self.assertEqual(stats["inline_math_tags_converted"], 1)

    def test_header_fallback_table_and_error_policy(self):
        table = "<table><tr><td><math>x</math></td></tr></table>"
        text, _ = serialize_page(
            {
                "blocks": [
                    {"html": "<p>Heading</p>", "label": "SectionHeader"},
                    {"html": table},
                    {"html": "not usable", "error": True},
                    {"html": "not usable", "skipped": True},
                    {"html": "repeated"},
                    {"html": "repeated"},
                ]
            }
        )
        self.assertEqual(text, "### Heading\n\n" + table + "\n\nrepeated\n\nrepeated")

    def test_incomplete_math_is_not_repaired(self):
        text, _ = serialize_page({"blocks": [{"html": '<math display="block">x'}]})
        self.assertEqual(text, '<math display="block">x')


class ReproductionIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"

    def test_identity_preserves_bytes_and_audit(self):
        original = b"repeat\r\nrepeat\r\n$$x$$\r\n"
        (self.source / "a.md").write_bytes(original)
        audit = prepare(self.source, self.output, "identity-v1", 1)
        self.assertEqual((self.output / "a.md").read_bytes(), original)
        self.assertEqual(audit["source"], audit["predictions"])
        self.assertEqual(audit["changed_markdown_files"], [])
        self.assertFalse(audit["ground_truth_used"])
        self.assertEqual(
            json.loads((self.output / "preparation.json").read_text()), audit
        )

    def test_display_profile_and_fingerprint(self):
        (self.source / "a.md").write_text("$$x$$ (1)\n", encoding="utf-8")
        audit = prepare(self.source, self.output, "display-math-v1", 1)
        self.assertEqual((self.output / "a.md").read_text(), "$$x$$\n\n(1)\n")
        self.assertEqual(audit["changed_markdown_files"], ["a.md"])
        digest = hashlib.sha256(
            b"a.md\0" + hashlib.sha256(b"$$x$$\n\n(1)\n").digest() + b"\n"
        ).hexdigest()
        self.assertEqual(audit["predictions"]["collection_sha256"], digest)

    def test_display_profile_retains_historical_newline_reading(self):
        (self.source / "a.md").write_bytes(b"$$x$$\r\n(1)\r\n")
        prepare(self.source, self.output, "display-math-v1", 1)
        self.assertEqual((self.output / "a.md").read_bytes(), b"$$x$$\n\n(1)\n")

    def test_surya_profile_and_counts(self):
        (self.source / "a.json").write_text(
            json.dumps({"blocks": [{"html": "<math>x</math>"}, {"error": True}]})
        )
        audit = prepare(self.source, self.output, "surya-html-v1", 1)
        self.assertEqual((self.output / "a.md").read_text(), "$x$\n")
        self.assertEqual(audit["serializer_counts"]["skipped_or_error_blocks"], 1)

    def test_refuses_existing_output_and_overlap(self):
        (self.source / "a.md").write_text("x")
        self.output.mkdir()
        (self.output / "keep").write_text("keep")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output, "identity-v1", 1)
        self.assertEqual((self.output / "keep").read_text(), "keep")
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / "nested", "identity-v1", 1)

    def test_rejects_whitespace_or_wrong_count(self):
        (self.source / "a.md").write_text("   \n")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output, "identity-v1", 1)
        (self.source / "a.md").write_text("x")
        with self.assertRaises(ValueError):
            prepare(self.source, self.output, "identity-v1", 2)
        self.assertFalse(self.output.exists())

    def test_failed_serialization_publishes_nothing(self):
        (self.source / "a.json").write_text(json.dumps({"blocks": [{"html": "x"}]}))
        (self.source / "b.json").write_text(json.dumps({"blocks": [{"error": True}]}))
        with self.assertRaises(ValueError):
            prepare(self.source, self.output, "surya-html-v1", 2)
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["source"])

    def test_rejects_wrong_surya_schema(self):
        (self.source / "a.json").write_text(json.dumps({"choices": []}))
        with self.assertRaisesRegex(ValueError, "blocks list"):
            prepare(self.source, self.output, "surya-html-v1", 1)
        self.assertFalse(self.output.exists())

    def test_verify_hash_names_config_and_count(self):
        (self.source / "a.md").write_text("x")
        prepare(self.source, self.output, "identity-v1", 1)
        config = self.root / "config.yaml"
        config.write_text("skip_chemical: true\n")
        fingerprint = inventory(self.source, ".md")["collection_sha256"]
        report = verify(
            self.source,
            self.output,
            config,
            1,
            fingerprint,
            fingerprint,
            sha256(config),
        )
        self.assertEqual(report["scoring_verification"], "not_run")
        with self.assertRaises(ValueError):
            verify(
                self.source,
                self.output,
                config,
                1,
                fingerprint,
                "wrong",
                sha256(config),
            )
        (self.output / "a.md").rename(self.output / "b.md")
        with self.assertRaisesRegex(ValueError, "page names differ"):
            verify(
                self.source,
                self.output,
                config,
                1,
                fingerprint,
                inventory(self.output, ".md")["collection_sha256"],
                sha256(config),
            )

    def test_cli_failure_does_not_leave_output(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "reproduction.prepare",
                "--source",
                str(self.source),
                "--output",
                str(self.output),
                "--profile",
                "identity-v1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preparation failed", result.stderr)
        self.assertFalse(self.output.exists())


class ManifestTests(unittest.TestCase):
    def test_manifest_and_frozen_config_agree(self):
        root = Path(__file__).resolve().parents[1] / "reproduction"
        manifest = json.loads((root / "historical-inputs.json").read_text())
        self.assertEqual(
            sha256(root / "scoring.yaml"), manifest["evaluator"]["config_sha256"]
        )
        self.assertEqual(len(manifest["models"]), 9)
        self.assertFalse(manifest["evaluator"]["scoring_changed"])
        self.assertIsNone(manifest["evaluator"]["public_image_reference"])
        for model in manifest["models"].values():
            self.assertEqual(len(model["prediction_collection_sha256"]), 64)
            self.assertIsNone(model["public_prediction_artifact"])
        self.assertEqual(
            manifest["models"]["paddleocr-vl-1.6"]["preparation_profile"], "identity-v1"
        )


if __name__ == "__main__":
    unittest.main()
