import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter

def plot_efficient_frontier(efficient_frontier, min_vol_portfolio=None, max_sharpe_portfolio=None, 
                           title='Efficient Frontier', filename=None, show_assets=False, 
                           asset_returns=None, asset_volatilities=None, asset_names=None):
    """
    Plot the efficient frontier with optional portfolio points.
    
    Parameters:
    -----------
    efficient_frontier : pandas.DataFrame
        DataFrame containing 'Return' and 'Volatility' columns
    min_vol_portfolio : dict, optional
        Dictionary with expected_return and volatility of minimum volatility portfolio
    max_sharpe_portfolio : dict, optional
        Dictionary with expected_return and volatility of maximum Sharpe ratio portfolio
    title : str, optional
        Plot title
    filename : str, optional
        If provided, save the plot to this file
    show_assets : bool, optional
        Whether to plot individual assets
    asset_returns : array-like, optional
        Expected returns of individual assets
    asset_volatilities : array-like, optional
        Volatilities of individual assets
    asset_names : list, optional
        Names of individual assets
    """
    plt.figure(figsize=(12, 8))
    
    # Format the axes
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    
    # Plot efficient frontier
    plt.scatter(efficient_frontier['Volatility'], efficient_frontier['Return'], 
                c=efficient_frontier['Sharpe'], cmap='viridis', s=30,
                edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cb = plt.colorbar()
    cb.set_label('Sharpe Ratio')
    
    # Plot minimum volatility portfolio if provided
    if min_vol_portfolio is not None:
        plt.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['expected_return'], 
                   marker='*', color='red', s=300, label='Minimum Volatility')
    
    # Plot maximum Sharpe ratio portfolio if provided
    if max_sharpe_portfolio is not None:
        plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], 
                   marker='D', color='green', s=200, label='Maximum Sharpe')
    
    # Plot individual assets if requested
    if show_assets and asset_returns is not None and asset_volatilities is not None:
        plt.scatter(asset_volatilities, asset_returns, marker='o', color='black', s=100)
        
        # Add asset labels if provided
        if asset_names is not None:
            for i, name in enumerate(asset_names):
                plt.annotate(name, (asset_volatilities[i], asset_returns[i]), 
                            xytext=(10, 0), textcoords='offset points')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Volatility (Standard Deviation)', fontsize=14)
    plt.ylabel('Expected Return', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save to file if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_weights(weights, title='Portfolio Weights', filename=None, sort=True):
    """
    Plot a bar chart of portfolio weights.
    
    Parameters:
    -----------
    weights : pandas.Series or dict
        Portfolio weights with asset names as index/keys
    title : str, optional
        Plot title
    filename : str, optional
        If provided, save the plot to this file
    sort : bool, optional
        Whether to sort weights by value
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    
    if sort:
        weights = weights.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    colors = cm.viridis(np.linspace(0, 1, len(weights)))
    weights.plot(kind='bar', color=colors)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Assets', fontsize=14)
    plt.ylabel('Weight', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Save to file if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()