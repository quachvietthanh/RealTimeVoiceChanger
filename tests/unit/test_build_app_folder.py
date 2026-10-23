"""Unit tests for scripts/build_app_folder.py on a temporary repository tree (no Python runtime, GPU or model load)."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
src_path = REPO_ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

_spec = importlib.util.spec_from_file_location("build_app_folder", REPO_ROOT / "scripts" / "build_app_folder.py")
assert _spec is not None and _spec.loader is not None
build_app_folder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_app_folder)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class BuildAppFolderTests(unittest.TestCase):
    CACHE = "third_party/applio/logs"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.out = Path(self.tmp.name) / "out"
        files = {
            "pyproject.toml": b"[project]\n", "README.md": b"readme", "config/README.md": b"cfg",
            "requirements/inference-win-py311-cu121.txt": b"torch==2.5.1+cu121\n",
            "third_party/applio/LICENSE": b"MIT", "third_party/applio/core.py": b"# gradio front end",
            "third_party/applio/rvc/infer/infer.py": b"# infer", "third_party/applio/rvc/configs/v2/40000.json": b"{}",
            "third_party/applio/rvc/models/pretraineds/f0G40k.pth": b"pretrained G",
            "third_party/applio/rvc/infer/__pycache__/infer.cpython-311.pyc": b"bytecode",
            f"{self.CACHE}/exp/G_2333333.pth": b"resume checkpoint",
            f"{self.CACHE}/exp/b.pth": b"model-b", f"{self.CACHE}/exp/b.index": b"index-b",
            "src/voice_changer/__init__.py": b"__version__ = '0.1.0'\n",
            "src/voice_changer/gui/assets/icons/app_icon.ico": b"ico",
            "src/voice_changer/__pycache__/x.pyc": b"bytecode",
            "models/checkpoints/rmvpe.pt": b"rmvpe", "models/checkpoints/contentvec/pytorch_model.bin": b"cv",
            "models/checkpoints/contentvec/config.json": b"{}",
            "models/weights/a.pth": b"model-a", "models/indices/a.index": b"index-a",
            "models/weights/j.pth": b"model-j", "models/weights/u.pth": b"model-u", "models/weights/s.pth": b"model-s",
            "data/held_out_test/x.wav": b"audio", "output/run/result.wav": b"audio",
        }
        for rel, content in files.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        registry = {"schema_version": 1, "default_profile": "a", "profiles": [
            {"id": "a", "display_name": "A", "status": "prototype", "model_path": "models/weights/a.pth",
             "model_sha256": sha(b"model-a"), "index_path": "models/indices/a.index", "index_sha256": sha(b"index-a"),
             "data_rights": "licensed"},
            {"id": "b", "display_name": "B", "status": "preview", "model_path": f"{self.CACHE}/exp/b.pth",
             "index_path": f"{self.CACHE}/exp/b.index", "data_rights": "consented"},
            {"id": "j", "display_name": "J", "status": "preview", "model_path": "models/weights/j.pth",
             "data_rights": "unverified"},
            # No data_rights field (reads as 'unspecified'), like the voice_b_preview entry; and the explicit value.
            {"id": "u", "display_name": "U", "status": "preview", "model_path": "models/weights/u.pth"},
            {"id": "s", "display_name": "S", "status": "preview", "model_path": "models/weights/s.pth",
             "data_rights": "unspecified"},
        ]}
        (self.repo / "config" / "voice_profiles.json").write_text(json.dumps(registry), encoding="utf-8")

    def build(self, **kw):
        return build_app_folder.build(self.out, repo=self.repo, **kw)

    def shipped(self):
        return sorted(p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file())

    def test_ships_code_and_config_but_no_caches_training_files_or_user_data(self):
        manifest = self.build()
        files = self.shipped()
        for rel in ("src/voice_changer/__init__.py", "src/voice_changer/gui/assets/icons/app_icon.ico",
                    "config/voice_profiles.json", "third_party/applio/rvc/infer/infer.py",
                    "third_party/applio/rvc/configs/v2/40000.json", "third_party/applio/LICENSE",
                    "requirements/inference-win-py311-cu121.txt", "BUILD_MANIFEST.json"):
            self.assertIn(rel, files)
        for marker in ("logs/", "pretraineds", "__pycache__", "core.py", "data/", "output/", "models/"):
            self.assertFalse([f for f in files if marker in f], marker)
        self.assertEqual(manifest["version"], build_app_folder.__version__)
        self.assertEqual(manifest["options"], {"with_aux_models": False, "profiles": []})
        self.assertIs(manifest["redistributable"], True)
        recorded = {f["path"]: f for f in manifest["files"]}
        self.assertEqual(sorted(recorded), [f for f in files if f != "BUILD_MANIFEST.json"])
        ico = recorded["src/voice_changer/gui/assets/icons/app_icon.ico"]
        self.assertEqual((ico["size"], ico["sha256"]), (3, sha(b"ico")))

    def test_aux_models_are_placed_where_the_adapter_reads_them(self):
        self.build(with_aux_models=True)
        for rel, content in (("models/checkpoints/rmvpe.pt", b"rmvpe"),
                             ("third_party/applio/rvc/models/predictors/rmvpe.pt", b"rmvpe"),
                             ("third_party/applio/rvc/models/embedders/contentvec/pytorch_model.bin", b"cv"),
                             ("third_party/applio/rvc/models/embedders/contentvec/config.json", b"{}")):
            self.assertEqual((self.out / rel).read_bytes(), content, rel)
        self.assertFalse((self.out / "third_party/applio/rvc/models/pretraineds").exists())

    def test_profile_files_are_shipped_only_when_safe_to_distribute(self):
        self.build(profile_ids=["a"])
        self.assertEqual((self.out / "models/weights/a.pth").read_bytes(), b"model-a")
        self.assertEqual((self.out / "models/indices/a.index").read_bytes(), b"index-a")
        for pid, message in (("b", "training cache"), ("j", "data rights 'unverified'"),
                             ("u", "data rights 'unspecified'"), ("s", "data rights 'unspecified'")):
            out = Path(self.tmp.name) / f"out_{pid}"
            with self.assertRaisesRegex(build_app_folder.BuildError, message):
                build_app_folder.build(out, profile_ids=[pid], repo=self.repo)
            self.assertFalse(out.exists())
        # One unverified profile among verified ones still refuses the whole build.
        with self.assertRaisesRegex(build_app_folder.BuildError, "profile 'u'"):
            build_app_folder.build(Path(self.tmp.name) / "out_mixed", profile_ids=["a", "u"], repo=self.repo)
        self.assertFalse((Path(self.tmp.name) / "out_mixed").exists())

    def test_rights_override_is_recorded_as_a_local_test_build(self):
        manifest = self.build(profile_ids=["a"])
        self.assertEqual((manifest["redistributable"], manifest["profiles_without_verified_rights"]), (True, []))
        for pid in ("j", "u"):
            out = Path(self.tmp.name) / f"out_{pid}_local"
            manifest = build_app_folder.build(out, profile_ids=["a", pid], allow_unverified_rights=True, repo=self.repo)
            self.assertIs(manifest["redistributable"], False)
            self.assertEqual(manifest["profiles_without_verified_rights"], [pid])
            recorded = json.loads((out / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertIs(recorded["redistributable"], False)

    def test_refuses_non_empty_output_and_the_repository_itself(self):
        self.out.mkdir()
        (self.out / "keep.txt").write_text("user file", encoding="utf-8")
        with self.assertRaisesRegex(build_app_folder.BuildError, "not empty"):
            self.build()
        self.assertEqual(self.shipped(), ["keep.txt"])
        for target in (self.repo, self.repo.parent):
            with self.assertRaisesRegex(build_app_folder.BuildError, "must not be the repository"):
                build_app_folder.build(target, repo=self.repo)

    def test_missing_applio_or_aux_model_is_reported(self):
        (self.repo / "models/checkpoints/rmvpe.pt").unlink()
        with self.assertRaisesRegex(build_app_folder.BuildError, "auxiliary model missing"):
            self.build(with_aux_models=True)
        (self.repo / "third_party/applio/rvc/infer/infer.py").unlink()
        with self.assertRaisesRegex(build_app_folder.BuildError, "applio/rvc is missing"):
            build_app_folder.build(Path(self.tmp.name) / "out2", repo=self.repo)


class RepositoryRegistryGateTests(unittest.TestCase):
    """Read-only planning against the real registry: nothing is copied."""

    def test_no_profile_without_verified_rights_can_be_planned_for_shipping(self):
        from voice_changer.profiles.registration import VERIFIED_RIGHTS
        from voice_changer.profiles.registry import ProfileRegistry

        registry = ProfileRegistry.from_file(REPO_ROOT / "config" / "voice_profiles.json", REPO_ROOT)
        checked = 0
        for profile in registry.list_profiles():
            if profile.data_rights in VERIFIED_RIGHTS or profile.model_path is None or not profile.files_present():
                continue
            with self.assertRaisesRegex(build_app_folder.BuildError, "must not be distributed"):
                build_app_folder.plan_files(False, [profile.profile_id], False, REPO_ROOT)
            checked += 1
        if not checked:
            self.skipTest("no local profile with unverified rights and model files")


if __name__ == "__main__":
    unittest.main()
