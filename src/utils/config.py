# FILE: /portfolio-optimizer/portfolio-optimizer/src/utils/config.py

# Configuration settings and constants used throughout the project

class Config:
    DATA_SOURCE = "path/to/data/source"
    OUTPUT_DIR = "path/to/output/directory"
    RISK_FREE_RATE = 0.01  # Example risk-free rate
    MAX_PORTFOLIO_SIZE = 10  # Maximum number of assets in the portfolio
    OPTIMIZATION_METHOD = "Markowitz"  # Default optimization method
    BLACK_LITTERMAN_TAU = 0.05  # Tau parameter for Black-Litterman model
    BLACK_LITTERMAN_P = None  # Views matrix for Black-Litterman model
    BLACK_LITTERMAN_Q = None  # View returns for Black-Litterman model

def get_config():
    return Config()