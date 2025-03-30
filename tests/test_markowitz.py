import unittest
import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

def test_portfolio_return_calculation():
    """Test that portfolio return calculation is correct."""
    expected_returns = np.array([0.1, 0.2, 0.15])
    cov_matrix = np.array([
        [0.05, 0.01, 0.02],
        [0.01, 0.06, 0.03],
        [0.02, 0.03, 0.04]
    ])
    
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix)
    
    # Equal weight portfolio
    weights = np.array([1/3, 1/3, 1/3])
    expected_return = np.sum(weights * expected_returns)
    
    assert np.isclose(optimizer.portfolio_return(weights), expected_return)

def test_portfolio_volatility_calculation():
    """Test that portfolio volatility calculation is correct."""
    expected_returns = np.array([0.1, 0.2, 0.15])
    cov_matrix = np.array([
        [0.05, 0.01, 0.02],
        [0.01, 0.06, 0.03],
        [0.02, 0.03, 0.04]
    ])
    
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix)
    
    # Equal weight portfolio
    weights = np.array([1/3, 1/3, 1/3])
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    assert np.isclose(optimizer.portfolio_volatility(weights), expected_volatility)

def test_minimum_volatility_optimization():
    """Test that minimum volatility optimization works."""
    expected_returns = np.array([0.1, 0.2, 0.15])
    cov_matrix = np.array([
        [0.05, 0.01, 0.02],
        [0.01, 0.06, 0.03],
        [0.02, 0.03, 0.04]
    ])
    
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix)
    result = optimizer.minimize_volatility()
    
    # Check that we have the expected keys
    assert 'weights' in result
    assert 'expected_return' in result
    assert 'volatility' in result
    
    # Check that weights sum to 1
    assert np.isclose(np.sum(result['weights']), 1.0)

if __name__ == '__main__':
    unittest.main()