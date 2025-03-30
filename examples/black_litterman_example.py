import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import DataLoader
from src.optimization.black_litterman import BlackLittermanModel
from src.visualization.efficient_frontier import plot_portfolio_weights

def main():
    # Define a list of assets
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
    
    print("Loading data...")
    # Load data for the past 5 years
    data_loader = DataLoader(symbols=symbols, start_date='2018-01-01')
    data_loader.load_data()
    
    # Calculate covariance matrix
    cov_matrix = data_loader.get_covariance_matrix()
    
    # Get market caps (we'll use fake data for demonstration)
    market_caps = np.array([
        2900, 2500, 1700, 1600, 900, 800, 700, 500, 450, 400
    ]) # in billions USD
    
    print("\nInitializing Black-Litterman model...")
    # Initialize the Black-Litterman model with risk aversion of 2.5
    bl_model = BlackLittermanModel(
        market_caps=market_caps,
        risk_aversion=2.5,
        cov_matrix=cov_matrix
    )
    
    # Print equilibrium returns
    equil_returns = pd.Series(bl_model.equil_returns, index=symbols)
    print("\nEquilibrium returns:")
    print(equil_returns)
    
    print("\nForming views on assets...")
    # Now let's form some views:
    # 1. We expect AAPL to outperform MSFT by 2%
    # 2. We expect GOOGL to have an absolute return of 15%
    # 3. We expect TSLA to underperform the market by 5%
    
    # Create the P matrix
    P = np.zeros((3, len(symbols)))
    P[0, symbols.index('AAPL')] = 1
    P[0, symbols.index('MSFT')] = -1
    P[1, symbols.index('GOOGL')] = 1
    P[2, symbols.index('TSLA')] = 1
    
    # Create the Q vector
    Q = np.array([0.02, 0.15, -0.05])
    
    # Define the uncertainty in our views
    omega = np.diag([0.01, 0.02, 0.03])  # Diagonal elements representing view uncertainty
    
    print("\nAdjusting market returns based on our views...")
    # Calculate posterior returns and optimal weights
    bl_result = bl_model.adjust_views(P, Q, omega)
    
    # Convert weights to Series for better display
    weights = pd.Series(bl_result['weights'], index=symbols)
    
    print("\nBlack-Litterman Portfolio:")
    print(f"Expected Return: {bl_result['expected_return']:.4f} ({bl_result['expected_return']*100:.2f}%)")
    print(f"Volatility: {bl_result['volatility']:.4f} ({bl_result['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {bl_result['expected_return']/bl_result['volatility']:.4f}")
    print("\nWeights:")
    print(weights)
    
    # Plot the portfolio weights
    print("\nPlotting Black-Litterman portfolio weights...")
    plot_portfolio_weights(
        weights,
        title='Black-Litterman Portfolio Weights',
        filename='black_litterman_weights.png'
    )

if __name__ == "__main__":
    main()