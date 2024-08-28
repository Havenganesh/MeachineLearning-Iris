from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

n = len(iris.feature_names)
fig, ax = plt.subplots(n, n, figsize=(16, 16))
colors = ["blue", "red", "green"]
for x in range(n):
    for y in range(n):
        xname = iris.feature_names[x]
        yname = iris.feature_names[y]
        for color_ind in range(len(iris.target_names)):
            ax[x, y].scatter(
                iris.data[iris.target == color_ind, x],
                iris.data[iris.target == color_ind, y],
                label=iris.target_names[color_ind],
                c=colors[color_ind],
            )
            ax[x, y].set_xlabel(xname)
            ax[x, y].set_ylabel(yname)

# ax[x, y].legend(loc='upper left')

plt.show()