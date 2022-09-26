# -*- coding: utf-8 -*-

import unittest
import numpy as np
from pythainlp.tools import misspell


def _count_difference(st1, st2):
    return sum(st1[i] != st2[i] for i in range(len(st1)))


class TestTextMisspellPackage(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "เรารักคุณมากที่สุดในโลก",
            "เราอยู่ที่มหาวิทยาลัยขอนแก่น"
        ]

    def test_misspell_naive(self):
        for text in self.texts:
            misspelled = misspell(text, ratio=0.1)

            self.assertEqual(len(text), len(misspelled))

            diff = _count_difference(text, misspelled)

            self.assertGreater(diff, 0, "we have some misspells.")

    def test_misspell_with_ratio_0_percent(self):
        for text in self.texts:
            misspelled = misspell(text, ratio=0.0)

            self.assertEqual(len(text), len(misspelled))

            diff = _count_difference(text, misspelled)

            self.assertEqual(
                diff, 0,
                "we shouldn't have any  misspell with ratio=0."
            )

    def test_misspell_with_ratio_50_percent(self):
        for text in self.texts:
            misspelled = misspell(text, ratio=0.5)

            self.assertEqual(len(text), len(misspelled))

            diff = _count_difference(text, misspelled)

            self.assertLessEqual(
                np.abs(diff - 0.5 * len(text)),
                2,
                f"expect 0.5*len(text)±2 misspells with ratio=0.5. (Δ={diff})",
            )

    def test_misspell_with_ratio_100_percent(self):
        for text in self.texts:
            misspelled = misspell(text, ratio=1)

            self.assertEqual(len(text), len(misspelled))

            diff = _count_difference(text, misspelled)

            self.assertLessEqual(
                np.abs(diff - len(text)),
                2,
                f"expect len(text)-2 misspells with ratio=1.5. (Δ={diff})",
            )
