import unittest
from src.optimization.markowitz import MarkowitzOptimizer

class TestMarkowitzOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = MarkowitzOptimizer()

    def test_calculate_optimal_weights(self):
        expected_weights = [0.4, 0.6]  # Example expected weights
        returns = [0.1, 0.2]  # Example returns
        risks = [0.15, 0.25]  # Example risks
        optimal_weights = self.optimizer.calculate_optimal_weights(returns, risks)
        self.assertAlmostEqual(optimal_weights[0], expected_weights[0], places=2)
        self.assertAlmostEqual(optimal_weights[1], expected_weights[1], places=2)

    def test_get_efficient_frontier(self):
        expected_frontier = [(0.1, 0.15), (0.2, 0.25)]  # Example expected frontier points
        frontier = self.optimizer.get_efficient_frontier()
        self.assertEqual(len(frontier), len(expected_frontier))
        for point in frontier:
            self.assertIn(point, expected_frontier)

if __name__ == '__main__':
    unittest.main()