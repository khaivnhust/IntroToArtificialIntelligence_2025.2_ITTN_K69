import unittest
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Ensure output encoding is utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.visual_feature_extract import VisualFeatureExtractor

class TestVisualFeatureExtractor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary dummy .npz file for testing."""
        cls.dummy_npz_path = PROJECT_ROOT / "test" / "dummy_features.npz"
        cls.feature_dim = 16 # Use a small dimension for testing
        
        # Create dummy data
        # Test both string keys and keys that need 0-padding
        cls.dummy_data = {
            "12345": np.random.rand(cls.feature_dim).astype(np.float32),
            "000054321": np.random.rand(cls.feature_dim).astype(np.float32)
        }
        
        np.savez(cls.dummy_npz_path, **cls.dummy_data)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up the dummy file after all tests are done."""
        if cls.dummy_npz_path.exists():
            os.remove(cls.dummy_npz_path)

    def setUp(self):
        """Initialize the extractor before each test."""
        self.extractor = VisualFeatureExtractor(
            npz_path=self.dummy_npz_path,
            feature_dimension=self.feature_dim
        )

    def test_load_features(self):
        """Test if features are successfully loaded from the NPZ file."""
        self.assertIn("12345", self.extractor._features)
        self.assertIn("000054321", self.extractor._features)
        
    def test_get_feature_vectors_existing(self):
        """Test retrieving features that exist in the dummy data."""
        # Note: '54321' will be padded to '000054321' internally
        article_ids = ["12345", 54321] 
        
        vectors = self.extractor.get_feature_vectors(article_ids)
        
        # Check shape and type
        self.assertIsInstance(vectors, torch.Tensor)
        self.assertEqual(vectors.shape, (2, self.feature_dim))
        
        # Check values
        np.testing.assert_allclose(vectors[0].numpy(), self.dummy_data["12345"], rtol=1e-5)
        np.testing.assert_allclose(vectors[1].numpy(), self.dummy_data["000054321"], rtol=1e-5)

    def test_get_feature_vectors_missing(self):
        """Test retrieving features for IDs that don't exist (should be zero vectors)."""
        article_ids = ["99999", "does_not_exist"]
        
        vectors = self.extractor.get_feature_vectors(article_ids)
        
        # Check shape and type
        self.assertIsInstance(vectors, torch.Tensor)
        self.assertEqual(vectors.shape, (2, self.feature_dim))
        
        # Check that they are all zeros
        self.assertTrue(torch.all(vectors == 0))
        
    def test_get_feature_vectors_mixed(self):
        """Test a mix of existing and missing IDs."""
        article_ids = ["12345", "missing_id", 54321]
        
        vectors = self.extractor.get_feature_vectors(article_ids)
        
        self.assertEqual(vectors.shape, (3, self.feature_dim))
        
        # 1st is present
        np.testing.assert_allclose(vectors[0].numpy(), self.dummy_data["12345"], rtol=1e-5)
        # 2nd is missing
        self.assertTrue(torch.all(vectors[1] == 0))
        # 3rd is present
        np.testing.assert_allclose(vectors[2].numpy(), self.dummy_data["000054321"], rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
