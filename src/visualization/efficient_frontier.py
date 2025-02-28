def plot_efficient_frontier(returns, risks, optimal_weights):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(risks, returns, marker='o', linestyle='-', color='b', label='Efficient Frontier')
    plt.scatter([sum(optimal_weights)], [sum(returns * optimal_weights)], color='r', marker='*', s=200, label='Optimal Portfolio')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid()
    plt.show()