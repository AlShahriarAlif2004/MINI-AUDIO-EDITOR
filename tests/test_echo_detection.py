import math
import unittest

import numpy as np

from backend.discrete_signal import Discrete_Signal


class EchoDetectionTests(unittest.TestCase):
    @staticmethod
    def _make_echo_signal(sample_rate=8000, duration=1.0, delay=0.11, decay=0.6):
        total_samples = int(sample_rate * duration)
        t = np.linspace(0.0, duration, total_samples, endpoint=False)
        carrier = np.sin(2 * math.pi * 220 * t)
        carrier += 0.25 * np.sin(2 * math.pi * 330 * t)
        noise = np.random.default_rng(0).normal(0.0, 0.12, size=total_samples)
        signal = carrier + noise
        echo_offset = int(delay * sample_rate)
        echo = np.zeros_like(signal)
        echo[echo_offset:] = signal[:-echo_offset] * decay
        return signal + echo

    def test_detect_echo_finds_expected_delay(self):
        sample_rate = 8000
        signal = self._make_echo_signal(sample_rate=sample_rate, delay=0.11, decay=0.6)
        result = Discrete_Signal(signal, sample_rate).detect_echo(min_delay=0.08, max_delay=0.3)

        delay_seconds, decay, confidence = result
        self.assertGreaterEqual(delay_seconds, 0.08)
        self.assertLessEqual(delay_seconds, 0.18)
        self.assertGreaterEqual(decay, 0.05)
        self.assertLessEqual(decay, 1.0)
        self.assertGreater(confidence, 0.1)

    def test_detect_echo_returns_no_result_for_clean_signal(self):
        sample_rate = 8000
        total_samples = int(sample_rate * 0.8)
        t = np.linspace(0.0, 0.8, total_samples, endpoint=False)
        noise = np.random.default_rng(1).normal(0.0, 0.12, size=total_samples)
        clean = np.sin(2 * math.pi * 220 * t) + 0.25 * np.sin(2 * math.pi * 330 * t) + noise

        delay_seconds, decay, confidence = Discrete_Signal(clean, sample_rate).detect_echo(min_delay=0.08, max_delay=0.3)

        self.assertEqual(delay_seconds, 0.0)
        self.assertEqual(decay, 0.0)
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
