def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std()

def calculate_volatility(returns):
    return returns.std()