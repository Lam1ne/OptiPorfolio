import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BlackLittermanModel:
    """Implementation of the Black-Litterman asset allocation model."""
    
    def __init__(self, market_caps, risk_aversion, cov_matrix, 
                 equil_returns=None, tau=0.025):
        """
        Initialize the Black-Litterman model.
        
        Parameters:
        -----------
        market_caps : array-like
            Market capitalizations of assets
        risk_aversion : float
            Risk aversion coefficient
        cov_matrix : ndarray or DataFrame
            Covariance matrix of asset returns
        equil_returns : array-like, optional
            Equilibrium returns (if None, will be calculated)
        tau : float, optional
            Scaling factor for estimation uncertainty
        """
        self.market_caps = np.array(market_caps, dtype=float)
        self.weights_market = self.market_caps / np.sum(self.market_caps)
        self.risk_aversion = risk_aversion
        self.cov_matrix = cov_matrix
        self.tau = tau
        
        if equil_returns is None:
            self.equil_returns = self.calculate_equilibrium_returns()
        else:
            self.equil_returns = np.array(equil_returns, dtype=float)
        
    def calculate_equilibrium_returns(self):
        """Calculate implied equilibrium returns using market weights."""
        return self.risk_aversion * self.cov_matrix.dot(self.weights_market)
    
    def incorporate_views(self, P, Q, omega=None):
        """
        Incorporate investor views into the model.
        
        Parameters:
        -----------
        P : ndarray
            Pick matrix for the views (each row corresponds to a view)
        Q : ndarray
            Expected returns for each view
        omega : ndarray, optional
            Uncertainty matrix for each view. If None, it will be calculated.
            
        Returns:
        --------
        ndarray
            Posterior expected returns
        """
        # Number of assets and views
        n_assets = len(self.equil_returns)
        n_views = len(Q)
        
        # If omega not provided, calculate it using the method in the paper
        if omega is None:
            omega = np.diag(np.dot(np.dot(P, self.cov_matrix), P.T)) * self.tau
            omega = np.diag(np.diag(omega))  # Just to ensure it's diagonal
        
        # Calculate posterior mean
        term1 = np.linalg.inv(np.linalg.inv(self.tau * self.cov_matrix) + 
                             np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        term2 = (np.dot(np.linalg.inv(self.tau * self.cov_matrix), self.equil_returns) + 
                np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))
        
        posterior_returns = np.dot(term1, term2)
        
        return posterior_returns
    
    def optimize_portfolio(self, expected_returns, cov_matrix=None):
        """
        Find the optimal portfolio weights given expected returns.
        
        Parameters:
        -----------
        expected_returns : ndarray
            Expected returns for each asset
        cov_matrix : ndarray, optional
            Covariance matrix (if None, use the one provided at initialization)
            
        Returns:
        --------
        ndarray
            Optimal portfolio weights
        """
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            port_return = np.sum(weights * expected_returns)
            port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            utility = port_return - 0.5 * self.risk_aversion * port_risk**2
            return -utility  # We minimize the negative utility
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result['x']
    
    def adjust_views(self, P, Q, omega=None):
        """
        Adjust views and find optimal portfolio weights.
        
        Parameters:
        -----------
        P : ndarray
            Pick matrix for the views
        Q : ndarray
            Expected returns for each view
        omega : ndarray, optional
            Uncertainty matrix for each view
            
        Returns:
        --------
        dict
            Dictionary containing optimal weights and portfolio statistics
        """
        posterior_returns = self.incorporate_views(P, Q, omega)
        optimal_weights = self.optimize_portfolio(posterior_returns)
        
        if hasattr(self.equil_returns, 'index'):
            # If we have asset names
            optimal_weights = pd.Series(optimal_weights, index=self.equil_returns.index)
        
        return {
            'weights': optimal_weights,
            'expected_return': np.sum(posterior_returns * optimal_weights),
            'volatility': np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
        }