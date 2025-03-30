import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzOptimizer:
    """Implementation of Markowitz's Modern Portfolio Theory."""
    
    def __init__(self, expected_returns, cov_matrix):
        """
        Initialize the MarkowitzOptimizer.
        
        Parameters:
        -----------
        expected_returns : pandas.Series or numpy.ndarray
            Expected returns for each asset
        cov_matrix : pandas.DataFrame or numpy.ndarray
            Covariance matrix of returns
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = expected_returns.index if isinstance(expected_returns, pd.Series) else None
        
    def portfolio_return(self, weights):
        """
        Calculate portfolio return.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Expected portfolio return
        """
        return np.sum(self.expected_returns * weights)
    
    def portfolio_volatility(self, weights):
        """
        Calculate portfolio volatility.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
            
        Returns:
        --------
        float
            Portfolio volatility (standard deviation)
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def minimize_volatility(self, target_return=None):
        """
        Find the portfolio weights that minimize volatility, 
        optionally subject to a target return constraint.
        
        Parameters:
        -----------
        target_return : float, optional
            Target portfolio return
            
        Returns:
        --------
        dict
            Dictionary containing optimal weights and portfolio statistics
        """
        num_assets = len(self.expected_returns)
        args = (self.cov_matrix,)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            args = (self.cov_matrix, self.expected_returns)
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_return(x) - target_return
            })
            
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array(num_assets * [1. / num_assets])
        
        result = minimize(
            fun=lambda w, c: np.sqrt(np.dot(w.T, np.dot(c, w))),
            x0=initial_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result['x']
        
        # Format the output as a dictionary
        output = {
            'weights': pd.Series(optimal_weights, index=self.asset_names) if self.asset_names is not None else optimal_weights,
            'expected_return': self.portfolio_return(optimal_weights),
            'volatility': self.portfolio_volatility(optimal_weights)
        }
        
        return output
    
    def efficient_frontier(self, points=20):
        """
        Calculate the efficient frontier.
        
        Parameters:
        -----------
        points : int
            Number of points to calculate
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing return, volatility, and Sharpe ratio for each point
        """
        # Get min and max returns for the range
        min_return = self.minimize_volatility()['expected_return']
        
        # Find maximum return portfolio (100% in the best performing asset)
        max_return_idx = np.argmax(self.expected_returns)
        max_return = self.expected_returns[max_return_idx]
        
        # Create range of target returns
        target_returns = np.linspace(min_return, max_return, points)
        efficient_portfolios = []
        
        # Calculate optimal portfolio for each target return
        for target in target_returns:
            efficient_portfolio = self.minimize_volatility(target_return=target)
            efficient_portfolios.append(efficient_portfolio)
        
        # Extract returns and volatilities
        returns = [p['expected_return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        sharpe_ratios = [r/v for r, v in zip(returns, volatilities)]
        
        # Create DataFrame with results
        return pd.DataFrame({
            'Return': returns,
            'Volatility': volatilities,
            'Sharpe': sharpe_ratios
        })