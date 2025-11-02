# ==========================
# 1️⃣ Install Libraries
# ==========================
!pip install pandas matplotlib seaborn scikit-learn ipywidgets --quiet

# ==========================
# 2️⃣ Import Libraries
# ==========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==========================
# 3️⃣ Upload Dataset
# ==========================
print("Upload your CSV file (e.g., Mall_Customers.csv)")
uploaded = files.upload()

for fn in uploaded.keys():
    df = pd.read_csv(fn)

print("Dataset Loaded Successfully!")
display(df.head())

# ==========================
# 4️⃣ Select Features
# ==========================
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# ==========================
# 5️⃣ Scale Features
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# 6️⃣ Interactive Clustering Function
# ==========================
def cluster_and_plot(k_clusters):
    clear_output(wait=True)
    print(f"Number of Clusters Selected: {k_clusters}")
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Plot clusters
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=features[0], y=features[1], 
                    hue='Cluster', palette='viridis', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='red', label='Centroids', marker='X')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Customer Segmentation using K-Means')
    plt.legend()
    plt.show()
    
    # Display cluster centers
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=features)
    print("Cluster Centers:")
    display(centroids_df)
    
    # Display cluster statistics
    for i in range(k_clusters):
        print(f"\nCluster {i} Stats:")
        display(df[df['Cluster']==i].describe())

# ==========================
# 7️⃣ Create Slider Widget
# ==========================
cluster_slider = widgets.IntSlider(value=4, min=2, max=10, step=1, description='Clusters:')
widgets.interact(cluster_and_plot, k_clusters=cluster_slider)
