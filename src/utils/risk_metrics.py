import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio of a portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
    risk_free_rate : float, optional
        Risk-free rate
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    expected_return = np.mean(returns)
    volatility = np.std(returns)
    
    return (expected_return - risk_free_rate) / volatility

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sortino ratio of a portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
    risk_free_rate : float, optional
        Risk-free rate
        
    Returns:
    --------
    float
        Sortino ratio
    """
    expected_return = np.mean(returns)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
    
    return (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else float('inf')

def calculate_maximum_drawdown(returns):
    """
    Calculate the maximum drawdown of a portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
        
    Returns:
    --------
    float
        Maximum drawdown
    """
    # Convert returns to cumulative returns
    cum_returns = (1 + pd.Series(returns)).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns / running_max) - 1
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def calculate_var(returns, confidence=0.05):
    """
    Calculate the Value at Risk (VaR) of a portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
    confidence : float, optional
        Confidence level (default 5%)
        
    Returns:
    --------
    float
        Value at Risk
    """
    return np.percentile(returns, confidence * 100)

def calculate_cvar(returns, confidence=0.05):
    """
    Calculate the Conditional Value at Risk (CVaR) of a portfolio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
    confidence : float, optional
        Confidence level (default 5%)
        
    Returns:
    --------
    float
        Conditional Value at Risk
    """
    var = calculate_var(returns, confidence)
    return np.mean(returns[returns <= var])