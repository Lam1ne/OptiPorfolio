# Portfolio Optimizer

This project implements a portfolio optimization tool based on Markowitz's Modern Portfolio Theory and advanced methods like the Black-Litterman model. It provides functionalities for data loading, portfolio optimization, risk assessment, and visualization of results.

## Features

- **Data Loading**: Load and preprocess market data using the `DataLoader` class.
- **Markowitz Optimization**: Calculate optimal portfolio weights and visualize the efficient frontier using the `MarkowitzOptimizer`.
- **Black-Litterman Model**: Adjust views and calculate weights with the `BlackLittermanModel`.
- **Risk Metrics**: Assess portfolio performance with functions to calculate Sharpe ratio and volatility.
- **Visualization**: Plot efficient frontiers and performance charts for better insights into portfolio performance.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Example of Markowitz Optimization

```python
from src.optimization.markowitz import MarkowitzOptimizer

optimizer = MarkowitzOptimizer()
optimal_weights = optimizer.calculate_optimal_weights()
```

### Example of Black-Litterman Model

```python
from src.optimization.black_litterman import BlackLittermanModel

bl_model = BlackLittermanModel()
adjusted_weights = bl_model.adjust_views()
```

## Running Tests

To run the unit tests, use:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.