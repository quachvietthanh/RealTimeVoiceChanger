"""RVCAdapter limits CPU thread pools for CUDA inference (idle pool threads spin-wait and burn CPU)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

from voice_changer.conversion.rvc import adapter as adapter_mod
from voice_changer.conversion.rvc.adapter import RVCAdapter, limit_cpu_threads


class CpuThreadTests(unittest.TestCase):
    def setUp(self):
        self.saved = torch.get_num_threads()

    def tearDown(self):
        limit_cpu_threads(self.saved)

    def test_limit_sets_torch_and_faiss(self):
        import faiss

        limit_cpu_threads(1)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(faiss.omp_get_max_threads(), 1)
        limit_cpu_threads(2)
        self.assertEqual(torch.get_num_threads(), 2)

    def test_load_model_limits_threads_only_for_cuda(self):
        with mock.patch.object(adapter_mod, "limit_cpu_threads") as limit, \
                mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(FileNotFoundError):  # stops right after the thread setup
                RVCAdapter(device="cuda").load_model("does/not/exist.pth")
            limit.assert_called_once_with(adapter_mod.CUDA_CPU_THREADS)
            limit.reset_mock()
            with self.assertRaises(FileNotFoundError):
                RVCAdapter(device="cpu").load_model("does/not/exist.pth")
            limit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
