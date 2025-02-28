import unittest
from src.optimization.black_litterman import BlackLittermanModel

class TestBlackLittermanModel(unittest.TestCase):

    def setUp(self):
        self.model = BlackLittermanModel()

    def test_adjust_views(self):
        views = {'AAPL': 0.05, 'GOOGL': 0.03}
        adjusted_views = self.model.adjust_views(views)
        self.assertIsInstance(adjusted_views, dict)

    def test_calculate_weights(self):
        expected_weights = self.model.calculate_weights()
        self.assertGreaterEqual(sum(expected_weights.values()), 0.99)
        self.assertLessEqual(sum(expected_weights.values()), 1.01)

if __name__ == '__main__':
    unittest.main()