"""Unit tests for registering a model for a planned profile (no GPU, no real models; inspection is mocked)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer.conversion.rvc.assets import CheckpointInfo, IndexInfo
from voice_changer.profiles.registration import (
    RegistrationError,
    format_registry,
    in_training_cache,
    profiles_in_training_cache,
    register_model,
    relocate_profile_assets,
)
from voice_changer.profiles.registry import DEFAULT_REGISTRY_PATH, ProfileRegistry

CKPT = CheckpointInfo(version="v2", f0=1, sample_rate=40000, vocoder="HiFi-GAN", embedder_dim=768,
                      speaker_count=1, embedder_model="contentvec")
IDX = IndexInfo(dim=768, ntotal=100, is_trained=True)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FormatTests(unittest.TestCase):
    def test_repository_registry_round_trips_byte_for_byte(self):
        raw = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        self.assertEqual(format_registry(json.loads(raw)), raw)


class RegisterModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "models").mkdir()
        for name, content in (("a.pth", b"model-a"), ("a.index", b"index-a"),
                              ("models/jett.pth", b"model-jett"), ("models/jett.index", b"index-jett")):
            (self.dir / name).write_bytes(content)
        self.data = {"schema_version": 1, "default_profile": "a", "profiles": [
            {"id": "a", "display_name": "A", "status": "preview", "model_path": "a.pth", "model_sha256": sha(b"model-a"),
             "index_path": "a.index", "index_sha256": sha(b"index-a"), "defaults": {"pitch_shift": 12.0}},
            {"id": "jett", "display_name": "Jett", "status": "planned", "data_rights": "unverified",
             "dataset": {"path": "data/speakers/jett"}, "defaults": {"pitch_shift": 12.0}},
            {"id": "proto", "display_name": "P", "status": "prototype", "model_path": "a.pth"},
        ]}
        self.path = self.dir / "voice_profiles.json"
        self.original = format_registry(self.data)
        self.path.write_text(self.original, encoding="utf-8")
        self.patches = [mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=CKPT),
                        mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IDX)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def register(self, profile="jett", model="models/jett.pth", index="models/jett.index", **kw):
        kw.setdefault("data_rights", "licensed")
        return register_model(profile, self.dir / model, self.dir / index if index else None,
                              registry_path=self.path, base_dir=self.dir, **kw)

    def assert_unchanged(self):
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["a.index", "a.pth", "models", "voice_profiles.json"])

    def test_dry_run_validates_and_writes_nothing(self):
        r = self.register()
        self.assertFalse(r.written)
        self.assertTrue(r.validation.ok)
        self.assertEqual((r.entry["status"], r.entry["model_path"], r.entry["model_sha256"], r.entry["index_path"]),
                         ("preview", "models/jett.pth", sha(b"model-jett"), "models/jett.index"))
        self.assert_unchanged()

    def test_write_changes_only_the_target_entry_and_keeps_a_backup(self):
        r = self.register(write=True)
        self.assertTrue(r.written)
        self.assertEqual(r.backup_path.read_text(encoding="utf-8"), self.original)
        new = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(new["profiles"][0], self.data["profiles"][0])  # A untouched
        self.assertEqual(new["profiles"][2], self.data["profiles"][2])
        jett = new["profiles"][1]
        self.assertEqual((jett["status"], jett["data_rights"], jett["dataset"]), ("preview", "licensed", {"path": "data/speakers/jett"}))
        reg = ProfileRegistry.from_file(self.path, base_dir=self.dir)
        self.assertTrue(reg.validate("jett").ok)
        self.assertFalse(any(p.suffix == ".tmp" for p in self.dir.iterdir()))

    def test_unverified_rights_need_an_explicit_flag(self):
        with self.assertRaisesRegex(RegistrationError, "data rights are 'unverified'"):
            self.register(data_rights=None)
        self.assert_unchanged()
        r = self.register(data_rights=None, allow_unverified_rights=True)
        self.assertEqual(r.entry["data_rights"], "unverified")
        self.assertTrue(any("unverified" in w for w in r.validation.warnings))

    def test_index_is_required_unless_explicitly_omitted(self):
        with self.assertRaisesRegex(RegistrationError, "index is required"):
            self.register(index=None)
        r = self.register(index=None, allow_no_index=True)
        self.assertNotIn("index_path", r.entry)

    def test_missing_files_are_refused(self):
        with self.assertRaisesRegex(RegistrationError, "model file not found"):
            self.register(model="models/nope.pth")
        with self.assertRaisesRegex(RegistrationError, "index file not found"):
            self.register(index="models/nope.index")
        self.assert_unchanged()

    def test_another_profiles_files_are_refused(self):
        with self.assertRaisesRegex(RegistrationError, "also registered for profile 'a'"):
            self.register(model="a.pth", index="a.index")
        (self.dir / "models/copy_of_a.pth").write_bytes(b"model-a")
        with self.assertRaisesRegex(RegistrationError, "same content as profile 'a'"):
            self.register(model="models/copy_of_a.pth")
        self.assert_unchanged()

    def test_training_cache_paths_are_refused(self):
        logs = self.dir / "third_party/applio/logs/x"
        logs.mkdir(parents=True)
        (logs / "jett.pth").write_bytes(b"model-jett-2")
        with self.assertRaisesRegex(RegistrationError, "training cache"):
            self.register(model="third_party/applio/logs/x/jett.pth")

    def test_incompatible_model_is_refused(self):
        bad = CheckpointInfo(**{**CKPT.__dict__, "embedder_model": "japanese-hubert-base"})
        with mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=bad):
            with self.assertRaisesRegex(RegistrationError, "embedder 'japanese-hubert-base'"):
                self.register()
        with mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IndexInfo(256, 100, True)):
            with self.assertRaisesRegex(RegistrationError, "index dimension 256"):
                self.register()
        self.assert_unchanged()

    def test_status_rules(self):
        with self.assertRaisesRegex(RegistrationError, "'prototype'"):
            self.register(profile="proto")
        with self.assertRaisesRegex(RegistrationError, "unknown voice profile"):
            self.register(profile="sage")
        self.register(write=True)
        with self.assertRaisesRegex(RegistrationError, "already has a model"):
            self.register()
        (self.dir / "models/jett2.pth").write_bytes(b"model-jett-2")
        r = self.register(model="models/jett2.pth", replace=True)
        self.assertEqual(r.entry["model_sha256"], sha(b"model-jett-2"))


class RelocateProfileAssetsTests(unittest.TestCase):
    """Repointing preview profiles that still read their files from third_party/applio/logs/."""

    CACHE = "third_party/applio/logs"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name, content in (("exp_a/a_50e.pth", b"model-a"), ("exp_a/added_a.index", b"index-a"),
                              ("exp_b/b_25e.pth", b"model-b"), ("exp_b/added_b.index", b"index-b")):
            path = self.dir / self.CACHE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.tuned = {"pitch_shift": 14.0, "f0_method": "rmvpe", "index_rate": 0.25, "filter_radius": 2,
                      "protect_rate": 0.4}
        self.data = {"schema_version": 1, "default_profile": "b", "profiles": [
            {"id": "a", "display_name": "A", "status": "preview",
             "model_path": f"{self.CACHE}/exp_a/a_50e.pth", "model_sha256": sha(b"model-a"),
             "index_path": f"{self.CACHE}/exp_a/added_a.index", "index_sha256": sha(b"index-a"),
             "defaults": {"pitch_shift": 12.0}, "provenance": {"experiment": "exp_a"},
             "withdrawn_reason": "unstable live playback"},
            {"id": "b", "display_name": "B", "status": "preview",
             "model_path": f"{self.CACHE}/exp_b/b_25e.pth", "model_sha256": sha(b"model-b"),
             "index_path": f"{self.CACHE}/exp_b/added_b.index", "index_sha256": sha(b"index-b"),
             "defaults": self.tuned, "provenance": {"experiment": "exp_b", "checkpoint": "25e"}},
            {"id": "unpinned", "display_name": "U", "status": "preview",
             "model_path": f"{self.CACHE}/exp_b/b_25e.pth"},
        ]}
        self.path = self.dir / "voice_profiles.json"
        self.original = format_registry(self.data)
        self.path.write_text(self.original, encoding="utf-8")
        self.cache_bytes = self._cache_snapshot()
        for p in (mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=CKPT),
                  mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IDX)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def _cache_snapshot(self):
        return {p.relative_to(self.dir).as_posix(): p.read_bytes()
                for p in sorted((self.dir / self.CACHE).rglob("*")) if p.is_file()}

    def relocate(self, profile="b", **kw):
        return relocate_profile_assets(profile, registry_path=self.path, base_dir=self.dir, **kw)

    def assert_cache_untouched(self):
        self.assertEqual(self._cache_snapshot(), self.cache_bytes)

    def test_lists_profiles_still_reading_from_the_training_cache(self):
        registry = ProfileRegistry.from_data(self.data, self.dir)
        self.assertEqual(profiles_in_training_cache(registry, self.dir), ["a", "b", "unpinned"])
        self.assertTrue(in_training_cache(self.dir / self.CACHE / "exp_b/b_25e.pth", self.dir))
        self.assertFalse(in_training_cache(self.dir / "models/weights/b_25e.pth", self.dir))
        self.assertFalse(in_training_cache(None, self.dir))

    def test_dry_run_plans_copies_and_changes_nothing(self):
        r = self.relocate()
        self.assertEqual([(s.kind, s.action, s.destination) for s in r.steps],
                         [("model", "copy", self.dir / "models/weights/b_25e.pth"),
                          ("index", "copy", self.dir / "models/indices/added_b.index")])
        self.assertFalse(r.written)
        self.assertIsNone(r.validation)
        self.assertFalse((self.dir / "models").exists())
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)
        self.assert_cache_untouched()

    def test_write_without_copies_is_refused(self):
        with self.assertRaisesRegex(RegistrationError, "copy the files first"):
            self.relocate(write=True)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_copy_then_write_repoints_only_the_paths(self):
        r = self.relocate(copy=True)
        self.assertEqual([s.action for s in r.steps], ["copied", "copied"])
        self.assertEqual((self.dir / "models/weights/b_25e.pth").read_bytes(), b"model-b")
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)  # copying alone never writes
        assert r.validation is not None
        self.assertTrue(r.validation.ok)

        r = self.relocate(write=True)
        self.assertEqual([s.action for s in r.steps], ["present", "present"])
        self.assertTrue(r.written)
        assert r.backup_path is not None
        self.assertEqual(r.backup_path.read_text(encoding="utf-8"), self.original)
        after = json.loads(self.path.read_text(encoding="utf-8"))
        before = self.data["profiles"]
        b = after["profiles"][1]
        self.assertEqual((b["model_path"], b["index_path"]), ("models/weights/b_25e.pth", "models/indices/added_b.index"))
        self.assertEqual(b["provenance"], {"experiment": "exp_b", "checkpoint": "25e", "relocated_from": {
            "model_path": f"{self.CACHE}/exp_b/b_25e.pth", "index_path": f"{self.CACHE}/exp_b/added_b.index"}})
        for key in set(b) - {"model_path", "index_path", "provenance"}:
            self.assertEqual(b[key], before[1][key], key)  # status, hashes and the tuned defaults are kept
        self.assertEqual(after["profiles"][0], before[0])
        self.assertEqual(after["profiles"][2], before[2])
        self.assertEqual(after["default_profile"], "b")
        reloaded = ProfileRegistry.from_file(self.path, self.dir).get("b")
        self.assertEqual(reloaded.defaults.pitch_shift, 14.0)
        self.assertEqual(reloaded.defaults.index_rate, 0.25)
        self.assert_cache_untouched()

    def test_withdrawn_profile_stays_withdrawn(self):
        self.relocate(profile="a", copy=True, write=True)
        a = ProfileRegistry.from_file(self.path, self.dir).get("a")
        self.assertEqual(a.withdrawn_reason, "unstable live playback")
        self.assertEqual(a.status, "preview")
        self.assertEqual(a.model_path, self.dir / "models/weights/a_50e.pth")
        self.assert_cache_untouched()

    def test_existing_different_destination_is_never_overwritten(self):
        dest = self.dir / "models/weights/b_25e.pth"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"someone else's model")
        with self.assertRaisesRegex(RegistrationError, "not an identical copy"):
            self.relocate(copy=True, write=True)
        self.assertEqual(dest.read_bytes(), b"someone else's model")
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_source_changed_since_registration_is_not_copied(self):
        (self.dir / self.CACHE / "exp_b/b_25e.pth").write_bytes(b"retrained in place")
        with self.assertRaisesRegex(RegistrationError, "no longer matches its pinned SHA-256"):
            self.relocate(copy=True)
        self.assertFalse((self.dir / "models/weights/b_25e.pth").exists())

    def test_unpinned_profiles_and_bad_destinations_are_refused(self):
        with self.assertRaisesRegex(RegistrationError, "not pinned"):
            self.relocate(profile="unpinned", copy=True)
        with self.assertRaisesRegex(RegistrationError, "inside third_party/applio/logs"):
            self.relocate(model_dest=self.dir / self.CACHE / "exp_b/copy.pth", copy=True)
        with self.assertRaisesRegex(RegistrationError, "must be inside"):
            self.relocate(model_dest=Path(tempfile.gettempdir()) / "elsewhere.pth", copy=True)
        with self.assertRaisesRegex(RegistrationError, "unknown voice profile"):
            self.relocate(profile="nope")
        self.assertFalse((self.dir / "models").exists())

    def test_copy_failing_verification_leaves_no_partial_file(self):
        real = __import__("voice_changer.conversion.rvc.assets", fromlist=["sha256_file"]).sha256_file
        calls = []

        def corrupt_on_copy(path):
            calls.append(Path(path).name)
            return "0" * 64 if Path(path).name.endswith(".partial") else real(path)

        with mock.patch("voice_changer.conversion.rvc.assets.sha256_file", side_effect=corrupt_on_copy):
            with self.assertRaisesRegex(RegistrationError, "copied model does not match"):
                self.relocate(copy=True)
        self.assertIn("b_25e.pth.partial", calls)
        self.assertEqual(list((self.dir / "models/weights").iterdir()), [])


class RegistryWriteBytesTests(unittest.TestCase):
    """The backup is the original file byte for byte and the rewrite keeps its line endings (regression: text-mode
    writes on Windows turned an LF registry and its backup into CRLF, so a restore was not byte-exact)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        cache = self.dir / "third_party/applio/logs/exp_b"
        cache.mkdir(parents=True)
        files = {"a.pth": b"model-a", "a.index": b"index-a", "jett.pth": b"model-jett", "jett.index": b"index-jett",
                 "third_party/applio/logs/exp_b/b.pth": b"model-b", "third_party/applio/logs/exp_b/b.index": b"index-b"}
        for rel, content in files.items():
            (self.dir / rel).write_bytes(content)
        self.data = {"schema_version": 1, "default_profile": "b", "profiles": [
            {"id": "a", "display_name": "A", "status": "preview", "model_path": "a.pth", "model_sha256": sha(b"model-a"),
             "index_path": "a.index", "index_sha256": sha(b"index-a"), "defaults": {"pitch_shift": 12.0},
             "withdrawn_reason": "unstable"},
            {"id": "b", "display_name": "B", "status": "preview",
             "model_path": "third_party/applio/logs/exp_b/b.pth", "model_sha256": sha(b"model-b"),
             "index_path": "third_party/applio/logs/exp_b/b.index", "index_sha256": sha(b"index-b"),
             "defaults": {"pitch_shift": 14.0, "index_rate": 0.25, "filter_radius": 2, "protect_rate": 0.4}},
            {"id": "jett", "display_name": "Jett", "status": "planned", "data_rights": "unverified"},
        ]}
        self.path = self.dir / "voice_profiles.json"
        for p in (mock.patch("voice_changer.conversion.rvc.assets.inspect_checkpoint", return_value=CKPT),
                  mock.patch("voice_changer.conversion.rvc.assets.inspect_index", return_value=IDX)):
            p.start()
            self.addCleanup(p.stop)

    def write_original(self, newline: str) -> bytes:
        raw = format_registry(self.data).replace("\n", newline).encode("utf-8")
        self.path.write_bytes(raw)
        return raw

    def test_relocation_keeps_lf_and_backs_up_exact_bytes(self):
        original = self.write_original("\n")
        relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, copy=True)
        r = relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, write=True)
        assert r.backup_path is not None
        self.assertEqual(r.backup_path.read_bytes(), original)
        after = self.path.read_bytes()
        self.assertNotIn(b"\r", after)
        # Only B's two paths change, plus the provenance line appended after its last field (which gains a comma).
        before_lines = original.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        self.assertEqual(len(after_lines), len(before_lines) + 1)
        removed = [ln for ln in before_lines if ln not in after_lines]
        added = [ln for ln in after_lines if ln not in before_lines]
        self.assertEqual(sorted(removed), sorted([
            b'      "model_path": "third_party/applio/logs/exp_b/b.pth",\n',
            b'      "index_path": "third_party/applio/logs/exp_b/b.index",\n',
            b'      "defaults": {"pitch_shift": 14.0, "index_rate": 0.25, "filter_radius": 2, "protect_rate": 0.4}\n']))
        self.assertEqual(sorted(added), sorted([
            b'      "model_path": "models/weights/b.pth",\n',
            b'      "index_path": "models/indices/b.index",\n',
            b'      "defaults": {"pitch_shift": 14.0, "index_rate": 0.25, "filter_radius": 2, "protect_rate": 0.4},\n',
            b'      "provenance": {"relocated_from": {"model_path": "third_party/applio/logs/exp_b/b.pth", '
            b'"index_path": "third_party/applio/logs/exp_b/b.index"}}\n']))
        # Restoring the backup gives the original file back exactly.
        self.path.write_bytes(r.backup_path.read_bytes())
        self.assertEqual(sha(self.path.read_bytes()), sha(original))

    def test_crlf_registry_stays_crlf(self):
        original = self.write_original("\r\n")
        relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, copy=True)
        r = relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, write=True)
        assert r.backup_path is not None
        self.assertEqual(r.backup_path.read_bytes(), original)
        after = self.path.read_bytes()
        self.assertEqual(after.count(b"\r\n"), after.count(b"\n"))
        self.assertEqual(json.loads(after)["profiles"][0], self.data["profiles"][0])

    def test_register_model_keeps_lf_unrelated_entries_and_exact_backup(self):
        original = self.write_original("\n")
        r = register_model("jett", self.dir / "jett.pth", self.dir / "jett.index", registry_path=self.path,
                           base_dir=self.dir, data_rights="licensed", write=True)
        assert r.backup_path is not None
        self.assertEqual(r.backup_path.read_bytes(), original)
        after = self.path.read_bytes()
        self.assertNotIn(b"\r", after)
        # Every line up to the target entry is byte-identical; the parsed entries A and B are unchanged.
        prefix = original[:original.index(b'"id": "jett"')]
        self.assertTrue(after.startswith(prefix))
        new = json.loads(after)
        self.assertEqual(new["profiles"][:2], self.data["profiles"][:2])
        self.assertEqual(new["default_profile"], "b")

    def test_existing_backup_name_is_never_overwritten(self):
        original = self.write_original("\n")
        relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, copy=True)
        with mock.patch("voice_changer.profiles.registration.datetime") as dt:
            dt.now.return_value.strftime.return_value = "20260101-000000"
            clash = self.dir / "voice_profiles.json.bak-20260101-000000"
            clash.write_bytes(b"earlier backup")
            with self.assertRaisesRegex(RegistrationError, "already exists"):
                relocate_profile_assets("b", registry_path=self.path, base_dir=self.dir, write=True)
        self.assertEqual(clash.read_bytes(), b"earlier backup")
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
