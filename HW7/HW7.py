import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Load the dataset
df = pd.read_csv("Country-data.csv")

# Save the country names for later use
countries = df["country"]

# Select relevant numerical features for clustering
df = df[["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]]

# Convert to NumPy array and ensure float type
X = df.copy().to_numpy() * 1.0

# Standardize the data by dividing by standard deviation (no centering)
std = np.std(X, axis=0)
X /= std

# Use the Elbow method to find the optimal number of clusters
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, n_init=30).fit(X)  # Fit KMeans with i clusters
    inertia[i-1] = kmeans.inertia_  # Save the inertia (sum of squared distances)

# Plot inertia vs number of clusters to visualize the elbow
plt.plot(np.arange(1, 15), inertia)
plt.show()

# Apply KMeans with 4 clusters (chosen based on elbow plot)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Print final inertia and unnormalized cluster centers
print(kmeans.inertia_)
print(kmeans.cluster_centers_ * std)  # Rescale centers to original units

# Print average stats and countries in each cluster
for i in range(4):
    print(f"Average stats for cluster {i}: {kmeans.cluster_centers_[i] * std} -------")
    for country in countries[kmeans.labels_ == i]:
        print(country)
