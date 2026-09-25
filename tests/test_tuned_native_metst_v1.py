import unittest

import numpy as np

import run_tuned_native_metst_v1 as tuned


class TunedNativeTests(unittest.TestCase):
    def test_native_configuration_is_in_grid(self):
        self.assertIn(tuned.NATIVE_CONFIG, tuned.GRID)
        self.assertEqual(len(tuned.GRID), 2160)

    def test_choose_uses_training_counts_only(self):
        counts = {config: (0, 1, 1) for config in tuned.GRID}
        counts[tuned.NATIVE_CONFIG] = (1, 0, 0)
        self.assertEqual(tuned.choose(counts), tuned.NATIVE_CONFIG)

    def test_decode_interval_width(self):
        class PMF:
            @staticmethod
            def moving_average(values, width):
                return np.asarray(values)

            @staticmethod
            def local_maxima(values, threshold, distance):
                return np.asarray([5])

        record = {"score": np.arange(11), "emotion": np.zeros(11)}
        result = tuned.decode(record, 2, PMF, (1.0, 0.55, 1.0, 1.5))
        self.assertEqual((result[0]["onset"], result[0]["offset"]), (2, 8))


if __name__ == "__main__":
    unittest.main()
