import pandas as pd
# Importing our clustering algorithm : Agglomerative
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import math


data = pd.read_csv("unsupervised_machine_learning/Mall_Customers.csv")

data = data.drop("CustomerID", axis=1)
# convert Gender values into numerical
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
print(data)
model = AgglomerativeClustering(
    n_clusters=5, metric='euclidean', linkage='complete')
# Applying agglomerative algorithm with 5 clusters, using euclidean distance as a metric
clust_labels = model.fit_predict(data)
agglomerative = pd.DataFrame(clust_labels)
print(agglomerative)
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data['Annual Income (k$)'],
                     data["Spending Score (1-100)"], c=agglomerative[0], s=100)
ax.set_title("Agglomerative Clutering")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
plt.colorbar(scatter)
# plt.show()

plt.figure(figsize=(10, 7))
plt.title("Customer Dendrograms")
dende = shc.dendrogram(shc.linkage(data, method="complete"))
plt.show()
