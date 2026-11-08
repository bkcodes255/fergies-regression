"""The structural privacy rule of the web export: a public page's exporter can't reach the squad."""
import inspect
import unittest

from src.web_data.export import PRIVATE, PUBLIC


class ExportContract(unittest.TestCase):
    def test_public_exporters_take_no_entry_id(self):
        for name, build in PUBLIC.items():
            self.assertEqual(list(inspect.signature(build).parameters), ["conn", "season"], name)

    def test_private_exporters_take_the_entry_id(self):
        for name, build in PRIVATE.items():
            self.assertEqual(list(inspect.signature(build).parameters), ["conn", "season", "entry_id"], name)

    def test_a_page_is_public_or_private_never_both(self):
        self.assertFalse(set(PUBLIC) & set(PRIVATE))


if __name__ == "__main__":
    unittest.main()
