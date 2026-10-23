"""External bear-case guard — offline tests for brain/core/external_bear_guard.py.

One test per scenario: a valid external bear case passes; a missing bear_case,
a non-external source_type, a missing url and a missing source_date are each
rejected.
"""
from __future__ import annotations

import unittest

from brain.core.external_bear_guard import require_external_bear_case


def _ext_bear() -> dict:
    return {"source_type": "external", "url": "https://example.com/bear", "source_date": "2026-06-01"}


class RequireExternalBearCase(unittest.TestCase):

    def test_accepts_valid_external_bear_case(self):
        require_external_bear_case(_ext_bear(), context="test")

    def test_rejects_missing_bear_case(self):
        with self.assertRaises(ValueError):
            require_external_bear_case(None, context="test")

    def test_rejects_non_external_source_type(self):
        with self.assertRaises(ValueError):
            require_external_bear_case(
                {"source_type": "analysis", "url": "x", "source_date": "y"}, context="test")

    def test_rejects_missing_url(self):
        with self.assertRaises(ValueError):
            require_external_bear_case(
                {"source_type": "external", "url": "", "source_date": "y"}, context="test")

    def test_rejects_missing_source_date(self):
        with self.assertRaises(ValueError):
            require_external_bear_case(
                {"source_type": "external", "url": "x", "source_date": ""}, context="test")


if __name__ == "__main__":
    unittest.main()
