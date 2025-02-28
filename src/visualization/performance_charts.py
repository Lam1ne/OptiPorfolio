from matplotlib import pyplot as plt

def plot_performance_charts(portfolio_returns, benchmark_returns, title='Portfolio Performance'):
    plt.figure(figsize=(10, 6))
    
    plt.plot(portfolio_returns, label='Portfolio Returns', color='blue')
    plt.plot(benchmark_returns, label='Benchmark Returns', color='orange')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid()
    plt.show()