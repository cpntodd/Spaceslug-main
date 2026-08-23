import unittest

from spaceslug.parity import compare_float_arrays


class ParityTest(unittest.TestCase):
    def test_pass_and_failure_are_structured(self):
        self.assertEqual(compare_float_arrays([1.0, 2.0], [1.0, 2.0])["status"], "pass")
        result = compare_float_arrays([1.0, 4.0], [1.0, 2.0], tolerance=0.1)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["first_bad_index"], 1)


if __name__ == "__main__":
    unittest.main()
