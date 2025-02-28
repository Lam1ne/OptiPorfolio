def calculate_correlation(data1, data2):
    """Calculate the correlation coefficient between two datasets."""
    return np.corrcoef(data1, data2)[0, 1]

def normalize_data(data):
    """Normalize the dataset to have a mean of 0 and a standard deviation of 1."""
    return (data - np.mean(data)) / np.std(data)