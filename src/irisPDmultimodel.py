import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the Iris data
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Plot a scatter matrix, with points color-coded by the iris target
pd.plotting.scatter_matrix(iris_df, c=iris.target, figsize=(8, 8), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8)

# Show the plot
plt.show()
