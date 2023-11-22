import pandas as pd
import numpy as np
from scipy import stats
# Importing our clustering algorithm : Agglomerative
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load the data in dataframe object
cards_data = pd.read_csv(
    'unsupervised_machine_learning/Credit_card_dataset.csv')
# getting some infos from Data
cards_data.info()
print(cards_data.head())

# I have an empty cell in CREDIT_LIMIT variable. Wee fill it with mean because it's numerical

cards_data['CREDIT_LIMIT'].fillna(
    cards_data['CREDIT_LIMIT'].mean(), inplace=True)
# We don't need the 'CUST_ID' in the classification then we drop it
cards_data = cards_data.drop('CUST_ID', axis=1)
print(cards_data.head())
print(cards_data.isnull().sum())

# I have made two classes one for the hierarchical and the other for the k-means
# clustering it will be easier for me to tweak he resolt


class Hierarchical:
    def __init__(self,  df):
        self.df = df
    # zcore methode

    def get_zscore(self):
        return np.abs(stats.zscore(self.df))

    # get the number of outliers
    def number_of_outlier(self, zscore_threshold):
        self.zscore_threshold = zscore_threshold
        z_scores = self.get_zscore()
        outliers_count = (
            self.df[z_scores > self.zscore_threshold]).notnull().sum()
        print(len(self.df))
        return outliers_count

        # delete the outliers we can specify the zscore_threshold
    def delete_outliers(self, zscore_threshold):
        z_scores = self.get_zscore()

        # Create a boolean mask for outliers
        outlier_mask = np.abs(z_scores) > zscore_threshold

        # Use .loc to set values in the original DataFrame
        # Set outliers to NaN
        self.df.loc[outlier_mask.any(axis=1), :] = np.nan

        # Drop rows with NaN values
        self.df.dropna(inplace=True)

        return self.df

        # Plot using the AgglomerativeClustering
    def plot_Agg_data(self, axeX, axeY, scale):
        model = AgglomerativeClustering(
            n_clusters=5, metric='euclidean', linkage='complete')
        # Applying agglomerative algorithm with 5 clusters, using euclidean distance as a metric
        clust_labels = model.fit_predict(self.df)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Assign different markers and colors for each cluster
        markers = ['o', 's', 'D', '^', 'v']
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for clust_label in range(len(markers)):
            cluster_data = self.df[clust_labels == clust_label]
            ax.scatter(cluster_data[axeX], cluster_data[axeY],
                       # type: ignore
                       # type: ignore
                       # type: ignore
                       # type: ignore
                       # type: ignore
                       # type: ignore
                       marker=markers[clust_label], color=colors[clust_label],
                       label=f'Cluster {clust_label}', s=50)

        ax.set_title("Agglomerative Clustering")
        ax.set_xlabel(axeX)
        ax.set_ylabel(axeY)
        ax.legend()
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        plt.show()

    def plot_Agg_data_viridis(self, axeX, axeY, cluster, scale):
        model = AgglomerativeClustering(
            n_clusters=cluster, metric='euclidean', linkage='complete')
        # Applying agglomerative algorithm with 5 clusters, using euclidean distance as a metric
        clust_labels = model.fit_predict(self.df)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Use cmap='viridis' and 'c=clust_labels' for coloring based on clusters
        scatter = ax.scatter(
            self.df[axeX], self.df[axeY], c=clust_labels, cmap='viridis', marker='o', s=25)

        ax.set_title("Agglomerative Clustering")
        ax.set_xlabel(axeX)
        ax.set_ylabel(axeY)
        # Add legend based on clusters
        ax.legend(*scatter.legend_elements(), title='Clusters')
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        # Add colorbar for cluster legend
        plt.colorbar(scatter, label='Clusters')
        plt.show()


class K_means:
    def __init__(self, df):
        self.df = df
        self.kmeans = None
        self.scaled_data = None
        self.optimal_k = None

    def get_zscore(self):
        return np.abs(stats.zscore(self.df))

    def delete_outliers(self, zscore_threshold):
        z_scores = self.get_zscore()

        # Create a boolean mask for outliers
        outlier_mask = np.abs(z_scores) > zscore_threshold

        # Use .loc to set values in the original DataFrame
        # Set outliers to NaN
        self.df.loc[outlier_mask.any(axis=1), :] = np.nan

        # Drop rows with NaN values
        self.df.dropna(inplace=True)

        return self.df

        # methode for plotting the elbow methode
    def elbow_method(self):
        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(self.df)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.show()

    def fit(self, optimal_k, scale_data=True):
        self.optimal_k = optimal_k
        self.cluster_column = 'Cluster'  # Store the cluster column name
        self.df_copy = self.df.copy()  # Create a copy for internal use

        if scale_data:
            self.scale_data()
            self.kmeans = KMeans(n_clusters=self.optimal_k,
                                 n_init=10, random_state=42)
            self.kmeans.fit(self.scaled_data)  # type: ignore
            self.df_copy[self.cluster_column] = self.kmeans.labels_
        else:
            self.kmeans = KMeans(n_clusters=self.optimal_k,
                                 n_init=10, random_state=42)
            self.kmeans.fit(self.df)
            self.df_copy[self.cluster_column] = self.kmeans.labels_

    def scale_data(self):
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.df)

    def plot_clusters(self, feature1, feature2, scale):
        predicted_labels = self.kmeans.predict(
            self.scaled_data if self.scaled_data is not None else self.df)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df[feature1], self.df[feature2],
                    c=predicted_labels, cmap='viridis', marker='o', edgecolor='black')
        plt.title('Clustered Data')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xscale(scale)
        plt.yscale(scale)
        plt.colorbar(label='Cluster')
        plt.show()


manip = Hierarchical(cards_data)
manip.delete_outliers(3)
manip.plot_Agg_data_viridis(
    axeX='PURCHASES', axeY='CREDIT_LIMIT', scale='log', cluster=3)


manip = K_means(cards_data)
manip.delete_outliers(zscore_threshold=3)
manip.elbow_method()
manip.fit(optimal_k=2, scale_data=True)
manip.plot_clusters('PURCHASES', 'CREDIT_LIMIT', scale='log')
# first i have deleted the outlier with threshold =3, i have the 2 plot class to classify the data into
# 5 cluster but it seems the graph are not goods and an group of clusters are notre very destinctive.
# the i have used the elbow methode in k)means it's tells me tha the best k is 2.
# with k=2 we can distinc very clearly the 2 classes of clusters
# same for the hiearchical clustering i have change it to 2 we can disting the clearly the 2 classes
# i have used the logarithmic scale to drow the graphics more clearly
