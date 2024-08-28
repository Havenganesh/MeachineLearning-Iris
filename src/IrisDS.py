from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.keys())
n_samples, n_features = iris.data.shape
print(iris.target_names)

fig, ax = plt.subplots()
x_index = 3
y_index = 0
colors = ["blue", "red", "green"]
for label, color in zip(range(len(iris.target_names)), colors):
    ax.hist(
        iris.data[iris.target == label, x_index],
        label=iris.target_names[label],
        color=color,
    )
ax.set_xlabel(iris.feature_names[x_index])
# ax.set_ylabel(iris.feature_names[y_index])
ax.legend(loc="upper right")
plt.show()
