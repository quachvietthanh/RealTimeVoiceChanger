"""Unit tests for the bundled asset manifest: provenance and license records, runtime gating of images,
packaging coverage and offline loading (headless; no network, audio device or model)."""

import fnmatch
import hashlib
import json
import socket
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.gui import presentation as pres
from voice_changer.gui.views.sidebar import logo_file_for

REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry(asset_id: str, path: str, license: str = "project_original", sha256=None, kind: str = "app_icon",
           attribution: str = "") -> dict:
    return {"id": asset_id, "path": path, "kind": kind, "used_for": "test", "source": "test", "license": license,
            "attribution": attribution, "verification": "test", "fallback": "monogram", "sha256": sha256}


class TestBundledManifest(unittest.TestCase):
    def test_bundled_manifest_records_every_asset_exactly(self) -> None:
        self.assertEqual(pres.verify_asset_manifest(), [])

    def test_every_image_the_gui_loads_is_redistributable(self) -> None:
        approved = pres.approved_assets()
        used = list(pres.icon_files())
        for size in (20, 24, 32, 48, 64):
            logo = logo_file_for(size)
            assert logo is not None, size
            used.append(Path(logo))
        used.append(pres.ICONS_DIR / "app_icon.ico")
        for path in used:
            rel = path.resolve().relative_to(pres.ASSETS_DIR).as_posix()
            self.assertIn(rel, approved, rel)

    def test_no_third_party_artwork_is_bundled(self) -> None:
        data = json.loads(pres.ASSET_MANIFEST_FILE.read_text(encoding="utf-8"))
        for e in data["assets"]:
            self.assertEqual(e["license"], "project_original", e["id"])
        excluded = {e["id"]: e for e in data["excluded"]}
        self.assertIn("valorant.agent_artwork", excluded)
        catalog = json.loads(pres.CHARACTER_CATALOG_FILE.read_text(encoding="utf-8"))
        for entry in catalog["entries"]:
            self.assertFalse({"avatar", "image", "portrait"} & set(entry), entry["id"])

    def test_package_data_ships_every_manifest_file(self) -> None:
        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            patterns = tomllib.load(f)["tool"]["setuptools"]["package-data"]["voice_changer.gui"]
        files = list(pres.load_asset_manifest()) + [pres.ASSET_MANIFEST_FILE.name]
        for rel in files:
            self.assertTrue(any(fnmatch.fnmatch(f"assets/{rel}", p) for p in patterns), rel)

    def test_manifest_and_presentation_load_without_network(self) -> None:
        def no_network(*args, **kwargs):
            raise AssertionError("network access attempted")

        with mock.patch.object(socket, "socket", no_network), \
                mock.patch.object(socket, "create_connection", no_network):
            self.assertEqual(pres.verify_asset_manifest(), [])
            self.assertTrue(pres.load_asset_manifest())
            self.assertTrue(pres.load_voice_presentation())
            self.assertTrue(pres.load_character_catalog().entries)


class TestManifestVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "icons").mkdir()
        self.png = b"\x89PNG fake icon"
        (self.dir / "icons" / "a.png").write_bytes(self.png)
        (self.dir / "data.json").write_text("{}", encoding="utf-8")
        self.entries = [_entry("icon.a", "icons/a.png", sha256=hashlib.sha256(self.png).hexdigest()),
                        _entry("data", "data.json", kind="data")]

    def verify(self) -> list[str]:
        manifest = self.dir / "asset_manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "assets": self.entries}), encoding="utf-8")
        return pres.verify_asset_manifest(manifest, self.dir)

    def test_complete_manifest_passes(self) -> None:
        self.assertEqual(self.verify(), [])

    def test_unlisted_file_is_reported(self) -> None:
        (self.dir / "icons" / "jett.png").write_bytes(b"dropped in")
        self.assertEqual(self.verify(), ["file is not recorded in the asset manifest: icons/jett.png"])

    def test_replaced_image_no_longer_matches_its_license_record(self) -> None:
        (self.dir / "icons" / "a.png").write_bytes(b"different artwork")
        self.assertTrue(any("SHA-256" in p for p in self.verify()))

    def test_missing_file_remote_path_and_escape_are_reported(self) -> None:
        self.entries += [_entry("gone", "icons/gone.png", sha256="0" * 64),
                         _entry("remote", "https://cdn.example/x.png"),
                         _entry("escape", "../outside.png")]
        problems = self.verify()
        self.assertTrue(any("missing" in p and "gone" in p for p in problems))
        self.assertTrue(any(p.startswith("remote:") for p in problems))
        self.assertTrue(any(p.startswith("escape:") for p in problems))

    def test_license_values_and_attribution_are_enforced(self) -> None:
        self.entries[0]["license"] = "found_online"
        self.assertTrue(any("license must be one of" in p for p in self.verify()))
        self.entries[0]["license"] = "licensed"
        self.assertTrue(any("attribution" in p for p in self.verify()))
        self.entries[0]["attribution"] = "Artist X, CC BY 4.0"
        self.assertEqual(self.verify(), [])

    def test_images_must_be_hashed_and_ids_unique(self) -> None:
        self.entries[0]["sha256"] = None
        self.entries[1]["id"] = "icon.a"
        problems = self.verify()
        self.assertTrue(any("only 'data' files may omit sha256" in p for p in problems))
        self.assertTrue(any("duplicate asset id icon.a" in p for p in problems))

    def test_unreadable_manifest_is_a_problem_not_a_crash(self) -> None:
        bad = self.dir / "asset_manifest.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
            problems = pres.verify_asset_manifest(bad, self.dir)
        self.assertIn("asset manifest is missing or unreadable", problems)


class TestRuntimeGating(unittest.TestCase):
    def test_png_without_a_redistributable_record_is_not_shown(self) -> None:
        unverified = pres.AssetRecord("x", "icons/app_icon_32.png", "avatar", "", "", "unverified", "", "", "", None)
        self.assertFalse(unverified.redistributable)
        with mock.patch.object(pres, "approved_assets", return_value={}):
            with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
                self.assertIsNone(pres._local_asset("icons/app_icon_32.png"))
        self.assertEqual(pres._local_asset("icons/app_icon_32.png"), pres.ICONS_DIR / "app_icon_32.png")

    def test_voice_avatar_pointing_at_an_unrecorded_image_falls_back_to_the_monogram(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "voice_presentation.json"
            path.write_text(json.dumps({"voices": {"jett": {"monogram": "JT", "color": "#be185d",
                                                            "avatar": "agents/jett.png"}}}), encoding="utf-8")
            with self.assertLogs("voice_changer.gui.presentation", level="WARNING"):
                table = pres.load_voice_presentation(path)
        self.assertIsNone(table["jett"].avatar.image)
        self.assertEqual(table["jett"].avatar.glyph, "JT")


if __name__ == "__main__":
    unittest.main()
