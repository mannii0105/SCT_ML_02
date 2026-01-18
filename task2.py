import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. LOAD DATA

# Assuming the file is named 'Mall_Customers.csv'
df = pd.read_csv('Mall_Customers.csv')


# 2. EXPLORATORY DATA ANALYSIS (EDA)

print(df.head())
print(df.info())

# We usually cluster based on 'Annual Income' and 'Spending Score' 
# to see the relationship between earnings and behavior.
X = df.iloc[:, [3, 4]].values 

# 3. THE ELBOW METHOD (Finding Optimal K)

wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_init=10, n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Based on the Elbow plot, k=5 is usually the optimal choice for this dataset.


# 4. TRAINING THE K-MEANS MODEL

kmeans = KMeans(n_init=10, n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster labels back to our original dataframe
df['Cluster'] = y_kmeans


# 5. VISUALIZING THE CLUSTERS

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], s=100, label='Cluster 1 (Sensible)')
sns.scatterplot(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], s=100, label='Cluster 2 (Standard)')
sns.scatterplot(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], s=100, label='Cluster 3 (Target/Rich)')
sns.scatterplot(x=X[y_kmeans == 3, 0], y=X[y_kmeans == 3, 1], s=100, label='Cluster 4 (Careless)')
sns.scatterplot(x=X[y_kmeans == 4, 0], y=X[y_kmeans == 4, 1], s=100, label='Cluster 5 (Frugal)')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids', marker='*')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# 6. EXPORT RESULTS

df.to_csv('Customer_Segments.csv', index=False)
print("Segmentation complete! Results saved to 'Customer_Segments.csv'")