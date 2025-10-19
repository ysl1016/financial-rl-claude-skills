import unittest
import importlib

class TestEnhancedProcessorRemoval(unittest.TestCase):
    """Verify that EnhancedDataProcessor is no longer available."""

    def test_module_is_missing(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("src.data.enhanced_processor")

    def test_attribute_not_exported(self):
        import src.data as data
        self.assertFalse(hasattr(data, "EnhancedDataProcessor"))

if __name__ == "__main__":
    unittest.main()
