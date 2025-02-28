# filepath: /portfolio-optimizer/portfolio-optimizer/src/main.py

from data.data_loader import DataLoader
from optimization.markowitz import MarkowitzOptimizer
from optimization.black_litterman import BlackLittermanModel
from visualization.efficient_frontier import plot_efficient_frontier
from visualization.performance_charts import plot_performance_charts

def main():
    # Load and preprocess market data
    data_loader = DataLoader()
    market_data = data_loader.load_data()
    preprocessed_data = data_loader.preprocess_data(market_data)

    # Optimize portfolio using Markowitz's theory
    markowitz_optimizer = MarkowitzOptimizer()
    optimal_weights = markowitz_optimizer.calculate_optimal_weights(preprocessed_data)
    efficient_frontier = markowitz_optimizer.get_efficient_frontier(preprocessed_data)

    # Visualize the efficient frontier
    plot_efficient_frontier(efficient_frontier)

    # Implement Black-Litterman model for portfolio adjustment
    black_litterman_model = BlackLittermanModel()
    adjusted_weights = black_litterman_model.adjust_views(optimal_weights)
    plot_performance_charts(adjusted_weights)

if __name__ == "__main__":
    main()