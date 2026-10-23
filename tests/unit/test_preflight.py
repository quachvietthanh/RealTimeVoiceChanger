"""Unit tests for the release preflight (temporary installation trees; no GPU, audio device or model load)."""

import hashlib
import json
import sys
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from voice_changer import main as cli
from voice_changer.diagnostics import preflight as pf


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class PreflightTreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = {
            "third_party/applio/rvc/infer/pipeline.py": b"class Pipeline:\n    def set_padding(self): pass\n",
            "models/checkpoints/rmvpe.pt": b"rmvpe",
            "models/checkpoints/contentvec/pytorch_model.bin": b"cv-weights",
            "models/checkpoints/contentvec/config.json": b"{}",
            "models/weights/a.pth": b"model-a",
            "models/indices/a.index": b"index-a",
            "third_party/applio/logs/exp/b.pth": b"model-b",
            "third_party/applio/logs/exp/b.index": b"index-b",
        }
        for rel, content in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.pinned = {"rmvpe.pt": sha(b"rmvpe"), "contentvec/pytorch_model.bin": sha(b"cv-weights"),
                       "contentvec/config.json": sha(b"{}")}
        self.registry = {"schema_version": 1, "default_profile": "a", "profiles": [
            {"id": "a", "display_name": "A", "status": "prototype", "model_path": "models/weights/a.pth",
             "model_sha256": sha(b"model-a"), "index_path": "models/indices/a.index", "index_sha256": sha(b"index-a")},
            {"id": "b", "display_name": "B", "status": "preview", "model_path": "third_party/applio/logs/exp/b.pth",
             "model_sha256": sha(b"model-b"), "index_path": "third_party/applio/logs/exp/b.index",
             "index_sha256": sha(b"index-b")},
            {"id": "c", "display_name": "C", "status": "preview", "model_path": "models/weights/missing.pth",
             "data_rights": "unverified"},
            {"id": "p", "display_name": "P", "status": "planned"},
            {"id": "u", "display_name": "U", "status": "preview", "model_path": "models/weights/a.pth",
             "data_rights": "unverified"},
        ]}
        self.write_registry()

    def write_registry(self):
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config" / "voice_profiles.json").write_text(json.dumps(self.registry), encoding="utf-8")

    def by_name(self, checks):
        return {c.name: c for c in checks}

    def test_profiles_report_cache_rights_pins_and_missing_files(self):
        checks = self.by_name(pf.check_profiles(self.root))
        self.assertEqual(checks["registry"].status, pf.PASS)
        self.assertEqual(checks["profile:a"].status, pf.PASS)
        self.assertEqual(checks["profile:b"].status, pf.WARN)
        self.assertIn("third_party/applio/logs/", checks["profile:b"].detail)
        self.assertEqual(checks["profile:c"].status, pf.WARN)
        self.assertIn("not on this machine", checks["profile:c"].detail)
        self.assertEqual(checks["profile:p"].status, pf.PASS)
        self.assertIn("not loadable", checks["profile:p"].detail)
        self.assertEqual(checks["profile:u"].status, pf.WARN)
        self.assertIn("do not distribute", checks["profile:u"].detail)
        self.assertIn("model SHA-256 not pinned", checks["profile:u"].detail)
        self.assertEqual(checks["default_profile"].status, pf.PASS)

    def test_hash_verification_catches_a_replaced_model(self):
        self.assertEqual(self.by_name(pf.check_profiles(self.root, verify_hashes=True))["profile:a"].status, pf.PASS)
        (self.root / "models/weights/a.pth").write_bytes(b"swapped")
        checks = self.by_name(pf.check_profiles(self.root, verify_hashes=True))
        self.assertEqual(checks["profile:a"].status, pf.FAIL)
        self.assertIn("model does not match", checks["profile:a"].detail)
        # Presence mode does not read the files, so it cannot see the swap.
        self.assertEqual(self.by_name(pf.check_profiles(self.root))["profile:a"].status, pf.PASS)

    def test_default_profile_without_files_fails(self):
        self.registry["default_profile"] = "c"
        self.write_registry()
        self.assertEqual(self.by_name(pf.check_profiles(self.root))["default_profile"].status, pf.FAIL)

    def test_unreadable_registry_fails_cleanly(self):
        (self.root / "config" / "voice_profiles.json").write_text("{", encoding="utf-8")
        checks = pf.check_profiles(self.root)
        self.assertEqual([(c.name, c.status) for c in checks], [("registry", pf.FAIL)])

    def test_auxiliary_assets_presence_and_hashes(self):
        self.assertTrue(all(c.status == pf.PASS for c in pf.check_auxiliary_assets(self.root)))
        verified = pf.check_auxiliary_assets(self.root, verify_hashes=True, pinned=lambda: self.pinned)
        self.assertTrue(all(c.status == pf.PASS and "verified" in c.detail for c in verified))
        (self.root / "models/checkpoints/rmvpe.pt").write_bytes(b"other")
        (self.root / "models/checkpoints/contentvec/config.json").unlink()
        checks = self.by_name(pf.check_auxiliary_assets(self.root, verify_hashes=True, pinned=lambda: self.pinned))
        self.assertEqual(checks["aux:rmvpe.pt"].status, pf.FAIL)
        self.assertIn("missing", checks["aux:contentvec/config.json"].detail)

    def test_auxiliary_hashes_unavailable_without_torch_are_reported(self):
        def no_torch():
            raise ModuleNotFoundError("No module named 'torch'")

        checks = pf.check_auxiliary_assets(self.root, verify_hashes=True, pinned=no_torch)
        self.assertTrue(all(c.status == pf.FAIL and "torch" in c.detail for c in checks))

    def test_applio_missing_or_unpatched(self):
        self.assertEqual(pf.check_applio(self.root).status, pf.PASS)
        (self.root / "third_party/applio/rvc/infer/pipeline.py").write_text("class Pipeline: pass\n", encoding="utf-8")
        self.assertIn("setup_applio_patch.py", pf.check_applio(self.root).detail)
        self.assertEqual(pf.check_applio(Path(self.tmp.name) / "nowhere").status, pf.FAIL)


class PreflightEnvironmentTests(unittest.TestCase):
    def test_missing_and_untested_packages(self):
        versions = {"torch": "2.6.0", "transformers": "5.1.0"}

        def find_spec(name):
            return None if name == "faiss" else object()

        def version_of(dist):
            if dist == "librosa":
                raise metadata.PackageNotFoundError(dist)
            return versions.get(dist, "1.0")

        checks = {c.name: c for c in pf.check_packages(find_spec, version_of)}
        self.assertEqual(checks["package:faiss-cpu"].status, pf.FAIL)
        self.assertEqual(checks["package:torch"].status, pf.WARN)
        self.assertEqual(checks["package:transformers"].status, pf.WARN)
        self.assertEqual(checks["package:numpy"].status, pf.PASS)
        self.assertEqual(checks["package:librosa"].detail, "unknown")
        # Imported at module level by the Applio modules the adapter loads (see docs/release-acceptance.md).
        for dist in ("torchcrepe", "noisereduce", "pedalboard", "pydub", "wget", "einops", "local-attention"):
            self.assertIn(f"package:{dist}", checks)

    def test_runtime_imports_of_applio_are_declared_and_checked(self):
        """Every third-party package imported at module level by the Applio modules the adapter loads must be a
        declared runtime dependency and a preflight check (regression: several were only in the train extra)."""
        import ast
        import re
        import tomllib

        repo = Path(__file__).resolve().parents[2]
        applio = repo / "third_party" / "applio"
        if not (applio / "rvc" / "infer" / "infer.py").is_file():
            self.skipTest("third_party/applio is not checked out")

        def source(mod):
            p = applio.joinpath(*mod.split("."))
            return next((c for c in (p.with_suffix(".py"), p / "__init__.py") if c.is_file()), None)

        seen, third, stack = set(), set(), ["rvc.infer.infer", "rvc.lib.utils", "rvc.lib.predictors.RMVPE"]
        while stack:
            mod = stack.pop()
            path = source(mod)
            if mod in seen or path is None:
                continue
            seen.add(mod)
            for node in ast.parse(path.read_text(encoding="utf-8", errors="replace")).body:
                for n in ast.walk(node) if isinstance(node, (ast.Try, ast.If)) else [node]:
                    names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                             [n.module] + [f"{n.module}.{a.name}" for a in n.names]
                             if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else [])
                    for name in names:
                        if source(name):
                            stack.append(name)
                        elif name.split(".")[0] not in sys.stdlib_module_names | {"rvc"}:  # rvc: namespace package
                            third.add(name.split(".")[0])
        self.assertGreater(len(seen), 5)
        with open(repo / "pyproject.toml", "rb") as f:
            declared = {re.split(r"[<>=!~ ;\[]", d, maxsplit=1)[0].lower()
                        for d in tomllib.load(f)["project"]["dependencies"]}
        checked = {module for module, _ in pf.RUNTIME_PACKAGES.values()}
        dist_of = {module: dist for dist, (module, _) in pf.RUNTIME_PACKAGES.items()}
        for module in sorted(third):
            self.assertIn(module, checked, f"{module} is not checked by the preflight")
            self.assertIn(dist_of[module], declared, f"{dist_of[module]} is not a declared runtime dependency")

    def test_constraints_file_pins_every_runtime_dependency_within_its_specifier(self):
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        repo = Path(__file__).resolve().parents[2]
        pins = {}
        for line in (repo / "requirements" / "inference-win-py311-cu121.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                name, _, version = line.partition("==")
                self.assertTrue(version, f"not an exact pin: {line!r}")
                pins[canonicalize_name(name)] = Version(version)
        with open(repo / "pyproject.toml", "rb") as f:
            deps = tomllib.load(f)["project"]["dependencies"]
        for dep in deps:
            req = Requirement(dep)
            pinned = pins.get(canonicalize_name(req.name))
            self.assertIsNotNone(pinned, f"{req.name} is not pinned in the constraints file")
            assert pinned is not None
            # "2.5.1+cu121" satisfies "==2.5.1" only once the local CUDA label is ignored.
            self.assertTrue(req.specifier.contains(Version(pinned.public), prereleases=True), f"{req.name}=={pinned}")

    def test_python_version(self):
        self.assertEqual(pf.check_python((3, 11, 9)).status, pf.PASS)
        self.assertEqual(pf.check_python((3, 12, 1)).status, pf.WARN)

    def test_report_counts_and_cli_exit_code(self):
        checks = [pf.Check("a", pf.PASS, "ok"), pf.Check("b", pf.WARN, "hm"), pf.Check("c", pf.FAIL, "bad")]
        self.assertTrue(pf.format_report(checks).endswith("1 passed, 1 warnings, 1 failed"))
        for result, code in ((checks, 1), (checks[:2], 0)):
            with mock.patch.object(sys, "argv", ["voice-changer", "--preflight"]), \
                    mock.patch("voice_changer.diagnostics.preflight.run_preflight", return_value=result) as run, \
                    mock.patch("builtins.print"):
                self.assertEqual(cli.main(), code)
            run.assert_called_once_with(verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
