import unittest

from spaceslug.backend import BackendSession
from spaceslug.inference_session import InferenceSession
from spaceslug.projected_attention_reference import ProjectedTinyAttentionModel


class InferenceSessionTest(unittest.TestCase):
    def test_cpu_session_records_backend_and_deterministic_token(self):
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(259))
        first = session.next_token([1, 2, 3])
        second = session.next_token([1, 2, 3])
        self.assertEqual(first, second)
        self.assertEqual(session.last_record["backend"], "cpu-reference")
        self.assertFalse(session.last_record["gpu_execution"])


if __name__ == "__main__":
    unittest.main()
