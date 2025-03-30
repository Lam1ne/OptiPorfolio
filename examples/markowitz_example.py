import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.optimization.markowitz import MarkowitzOptimizer
from src.visualization.efficient_frontier import plot_efficient_frontier, plot_portfolio_weights
from src.utils.risk_metrics import calculate_sharpe_ratio

def main():
    # Define a list of assets
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
    
    print("Loading data...")
    # Load data for the past 5 years
    data_loader = DataLoader(symbols=symbols, start_date='2018-01-01')
    data_loader.load_data()
    
    # Calculate expected returns and covariance matrix
    expected_returns = data_loader.get_annualized_returns()
    cov_matrix = data_loader.get_covariance_matrix()
    
    print("\nExpected Returns:")
    print(expected_returns)
    
    print("\nInitializing Markowitz optimizer...")
    # Initialize the optimizer
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix)
    
    print("\nCalculating minimum volatility portfolio...")
    # Find the minimum volatility portfolio
    min_vol_portfolio = optimizer.minimize_volatility()
    
    print("\nMinimum Volatility Portfolio:")
    print(f"Expected Return: {min_vol_portfolio['expected_return']:.4f} ({min_vol_portfolio['expected_return']*100:.2f}%)")
    print(f"Volatility: {min_vol_portfolio['volatility']:.4f} ({min_vol_portfolio['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {min_vol_portfolio['expected_return']/min_vol_portfolio['volatility']:.4f}")
    print("\nWeights:")
    print(min_vol_portfolio['weights'])
    
    print("\nCalculating efficient frontier...")
    # Calculate the efficient frontier
    efficient_frontier = optimizer.efficient_frontier(points=50)
    
    # Find the maximum Sharpe ratio portfolio
    max_sharpe_idx = efficient_frontier['Sharpe'].idxmax()
    max_sharpe_portfolio = {
        'Volatility': efficient_frontier.loc[max_sharpe_idx, 'Volatility'],
        'Return': efficient_frontier.loc[max_sharpe_idx, 'Return'],
        'Sharpe': efficient_frontier.loc[max_sharpe_idx, 'Sharpe']
    }
    
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"Expected Return: {max_sharpe_portfolio['Return']:.4f} ({max_sharpe_portfolio['Return']*100:.2f}%)")
    print(f"Volatility: {max_sharpe_portfolio['Volatility']:.4f} ({max_sharpe_portfolio['Volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.4f}")
    
    # Get individual asset volatilities for plotting
    asset_volatilities = np.sqrt(np.diag(cov_matrix))
    
    # Plot the efficient frontier
    print("\nPlotting efficient frontier...")
    plot_efficient_frontier(
        efficient_frontier,
        min_vol_portfolio,
        max_sharpe_portfolio,
        show_assets=True,
        asset_returns=expected_returns.values,
        asset_volatilities=asset_volatilities,
        asset_names=symbols,
        title='Efficient Frontier - Markowitz Optimization',
        filename='markowitz_efficient_frontier.png'
    )
    
    # Plot the portfolio weights
    print("\nPlotting minimum volatility portfolio weights...")
    plot_portfolio_weights(
        min_vol_portfolio['weights'],
        title='Minimum Volatility Portfolio Weights',
        filename='min_vol_weights.png'
    )

if __name__ == "__main__":
    main()