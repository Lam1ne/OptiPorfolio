from src.data.data_loader import DataLoader
from src.data.market_data import MarketData
from src.optimization.markowitz import MarkowitzOptimizer
from src.visualization.efficient_frontier import plot_efficient_frontier

# Load market data
data_loader = DataLoader()
market_data = MarketData()

# Assuming we have a method to load and preprocess data
price_data = data_loader.load_data('path_to_data.csv')
preprocessed_data = data_loader.preprocess_data(price_data)

# Set market data
market_data.set_price_data(preprocessed_data)

# Initialize the Markowitz optimizer
optimizer = MarkowitzOptimizer()

# Calculate optimal weights
optimal_weights = optimizer.calculate_optimal_weights(market_data.get_returns())

# Plot the efficient frontier
plot_efficient_frontier(optimal_weights, market_data.get_returns())