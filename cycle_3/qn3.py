import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets
import mpl_toolkits.mplot3d 

iris=pd.read_csv(r"/content/drive/MyDrive/Iris.csv")
print(iris)

ax=plt.subplots(1,1,figsize=(5,5))
sns.countplot(data=iris,x="Species")
plt.title("Iris Species Count")
plt.show()
print("\n")


np.random.seed(5)

iris1 = datasets.load_iris()
X = iris1.data
y = iris1.target

fig = plt.figure(1, figsize=(4, 4))
plt.clf()

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 0].mean(),
        X[y == label, 1].mean() + 1.5,
        X[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
print("3-Dimentsional Scatter Plot")
plt.show()
print("\n")

plt.figure(figsize = (8, 4))
x = iris["SepalLengthCm"]
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")


plt.figure(figsize = (7.93, 4))
x = iris["SepalWidthCm"]
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")

plt.figure(figsize = (8, 4))
x = iris["PetalLengthCm"]
plt.hist(x, bins = 20, color = "green")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")


plt.figure(figsize = (8, 4))
x = iris["PetalWidthCm"]
plt.hist(x, bins = 20, color = "green")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")
