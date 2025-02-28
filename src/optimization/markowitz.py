class MarkowitzOptimizer:
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = None
        self.cov_matrix = None

    def calculate_optimal_weights(self, risk_aversion):
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        ones = np.ones(len(self.mean_returns))
        weights = inv_cov_matrix @ (self.mean_returns - risk_aversion * ones)
        weights /= np.sum(weights)
        return weights

    def get_efficient_frontier(self, num_portfolios=10000):
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(self.mean_returns))
            weights /= np.sum(weights)
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std_dev
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe ratio
        return results