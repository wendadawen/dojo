from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".dojo" / "scripts"))

from catalog_builder import build_catalog  # noqa: E402
from migrate_page_metadata import migrate  # noqa: E402


class DojoSiteTest(unittest.TestCase):
    def test_real_catalog_covers_every_wiki_page(self) -> None:
        catalog = build_catalog(ROOT)
        expected = {
            page.relative_to(ROOT).as_posix()
            for page in (ROOT / "wiki").glob("*/index.html")
        }
        actual = {page["id"] for page in catalog["pages"]}
        self.assertEqual(actual, expected)
        self.assertEqual(catalog["warnings"], [])
        self.assertTrue(
            all(
                edge["source"] in actual and edge["target"] in actual
                for edge in catalog["edges"]
            )
        )

    def test_every_wiki_page_has_home_metadata(self) -> None:
        required = (
            'name="description"',
            'name="dojo:type"',
            'name="dojo:topics"',
            'name="dojo:tag"',
        )
        for page in (ROOT / "wiki").glob("*/index.html"):
            head = page.read_text(encoding="utf-8", errors="ignore").split("<body", 1)[0]
            self.assertTrue(all(item in head for item in required), page)

    def test_metadata_migration_preserves_body_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = root / "wiki" / "sample" / "index.html"
            page.parent.mkdir(parents=True)
            page.write_text(
                "<!DOCTYPE html>\n<html>\n<head>\n"
                '  <meta name="viewport" content="width=device-width">\n'
                "  <title>Sample</title>\n</head>\n"
                "<body><h1>Body</h1></body>\n</html>",
                encoding="utf-8",
            )
            manifest = root / "content.json"
            manifest.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "section": "concepts",
                                "path": "wiki/sample/index.html",
                                "tag": "测试",
                                "title": "Sample",
                                "desc": "Description",
                                "group": "主题",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            before_body = page.read_text(encoding="utf-8").split("<body", 1)[1]
            first = migrate(root, manifest, write=True)
            once = page.read_bytes()
            second = migrate(root, manifest, write=True)
            after_body = page.read_text(encoding="utf-8").split("<body", 1)[1]
            self.assertEqual(first["changed"], ["wiki/sample/index.html"])
            self.assertEqual(second["changed"], [])
            self.assertEqual(page.read_bytes(), once)
            self.assertEqual(before_body, after_body)

    def test_homepage_is_stable_shell_with_local_assets(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for element_id in (
            "search-input",
            "library-grid",
            "global-graph",
            "relation-panel",
            "local-graph",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertNotIn('href="wiki/kda/index.html"', html)
        for path in (
            "assets/dojo-home.css",
            "assets/dojo-home-model.js",
            "assets/dojo-graph.js",
            "assets/dojo-home.js",
        ):
            self.assertTrue((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
