import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset
iris = load_iris()

# Initialize an empty list to hold the data for each class
X = [[] for _ in range(3)]

# Separate the data points based on their class
for i in range(len(iris.data)):
    iclass = iris.target[i]
    X[iclass].append([iris.data[i][0], iris.data[i][1], sum(iris.data[i][2:])])

# Separate the features for each class
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each class
for iclass in range(3):
    xs = [x[0] for x in X[iclass]]
    ys = [x[1] for x in X[iclass]]
    zs = [x[2] for x in X[iclass]]
    ax.scatter(xs, ys, zs, c=colours[iclass])

# Display the plot
plt.show()
