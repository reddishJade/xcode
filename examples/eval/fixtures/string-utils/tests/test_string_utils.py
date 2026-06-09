import unittest

from string_utils import reverse_string, count_words


class TestStringUtils(unittest.TestCase):
    def test_reverse_string(self):
        self.assertEqual(reverse_string("hello"), "olleh")

    def test_count_words(self):
        self.assertEqual(count_words("hello world"), 2)


if __name__ == "__main__":
    unittest.main()
