import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class DataLoader:
    """Class for loading and processing financial data."""
    
    def __init__(self, symbols, start_date=None, end_date=None):
        """
        Initialize the DataLoader.
        
        Parameters:
        -----------
        symbols : list
            List of ticker symbols
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        """
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        
    def load_data(self):
        """Load historical price data from Yahoo Finance."""
        self.data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
        return self.data
    
    def calculate_returns(self, period='daily'):
        """
        Calculate returns from price data.
        
        Parameters:
        -----------
        period : str
            'daily' or 'monthly'
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame of returns
        """
        if self.data is None:
            self.load_data()
            
        if period == 'daily':
            returns = self.data.pct_change().dropna()
        elif period == 'monthly':
            returns = self.data.resample('M').last().pct_change().dropna()
        else:
            raise ValueError("Period must be 'daily' or 'monthly'")
            
        return returns
    
    def get_annualized_returns(self):
        """Calculate annualized returns based on daily returns."""
        daily_returns = self.calculate_returns(period='daily')
        return daily_returns.mean() * 252
    
    def get_covariance_matrix(self, period='daily'):
        """
        Calculate the covariance matrix of returns.
        
        Parameters:
        -----------
        period : str
            'daily' or 'monthly'
            
        Returns:
        --------
        pandas.DataFrame
            Covariance matrix
        """
        returns = self.calculate_returns(period=period)
        
        if period == 'daily':
            return returns.cov() * 252
        elif period == 'monthly':
            return returns.cov() * 12