import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from optimization.markowitz import MarkowitzOptimizer
from optimization.black_litterman import BlackLittermanModel
from visualization.efficient_frontier import plot_efficient_frontier
from visualization.performance_charts import plot_performance_charts

def main():
    # Define a sample portfolio of stocks
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH']
    
    # Load data
    data_loader = DataLoader(symbols=symbols, start_date='2018-01-01')
    data_loader.load_data()
    
    # Calculate expected returns and covariance matrix
    expected_returns = data_loader.get_annualized_returns()
    cov_matrix = data_loader.get_covariance_matrix()
    
    # Initialize the Markowitz optimizer
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix)
    
    # Get the minimum volatility portfolio
    min_vol_portfolio = optimizer.minimize_volatility()
    
    print("Minimum Volatility Portfolio:")
    print(f"Expected Return: {min_vol_portfolio['expected_return']:.4f}")
    print(f"Volatility: {min_vol_portfolio['volatility']:.4f}")
    print("Weights:")
    print(min_vol_portfolio['weights'])
    
    # Calculate the efficient frontier
    efficient_frontier = optimizer.efficient_frontier(points=50)
    
    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(efficient_frontier['Volatility'], efficient_frontier['Return'], 
                c=efficient_frontier['Sharpe'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['expected_return'], 
                marker='*', color='r', s=300, label='Minimum Volatility')
    
    # Find the maximum Sharpe ratio portfolio
    max_sharpe_idx = efficient_frontier['Sharpe'].idxmax()
    max_sharpe_portfolio = efficient_frontier.loc[max_sharpe_idx]
    plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], 
                marker='o', color='g', s=200, label='Maximum Sharpe')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig('efficient_frontier.png')
    plt.show()

if __name__ == "__main__":
    main()