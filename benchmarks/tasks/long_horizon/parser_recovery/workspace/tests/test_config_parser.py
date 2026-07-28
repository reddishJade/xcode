"""配置解析器行为测试。"""

from __future__ import annotations

import unittest

from src.public_api import ParseError, parse_config


class ParseConfigTests(unittest.TestCase):
    def test_trims_keys_and_values(self) -> None:
        self.assertEqual(
            parse_config(["  host = example.test  "]), {"host": "example.test"}
        )

    def test_value_may_contain_equals(self) -> None:
        self.assertEqual(
            parse_config(["endpoint=https://example.test/search?q=a=b"]),
            {"endpoint": "https://example.test/search?q=a=b"},
        )

    def test_ignores_blank_lines_and_comments(self) -> None:
        self.assertEqual(parse_config(["", "   # note", "port=8080"]), {"port": "8080"})

    def test_duplicate_key_uses_last_value(self) -> None:
        self.assertEqual(parse_config(["mode=old", "mode=new"]), {"mode": "new"})

    def test_rejects_missing_separator(self) -> None:
        with self.assertRaises(ParseError):
            parse_config(["not-a-setting"])

    def test_rejects_empty_key_after_trimming(self) -> None:
        with self.assertRaises(ParseError):
            parse_config(["   =value"])


if __name__ == "__main__":
    unittest.main()
