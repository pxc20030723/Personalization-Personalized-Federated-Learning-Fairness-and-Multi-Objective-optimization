'''
Condidering that 18 catagories of movies are too complex and there may exsits some relevance between these catagories

we try PCA as preparation for clustering
'''

import matplotlib.pyplot as plt
from statistics import covariance
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df=pd.read_csv("/Users/xinchepeng/Documents/Github_projects/18667_project/data/user_preferences_count.csv")

genere_columns=['Action', 'Adventure', 'Animation', "Children's",
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western']
X=df[genere_columns].values #(943,18)
mean=np.mean(X, axis=0)
std=np.std(X,axis=0)
epsilon=1e-8
X_scaled=(X-mean)/(std + epsilon)
'''
pca=PCA()
pca.fit(X_scaled)
eigenvalue=pca.explained_variance_ratio_
accumulated_eigenvaleu=np.cumsum(eigenvalue)
print(accumulated_eigenvaleu[:12])
'''
#choose k=8  80% accumulated_eigenvalue

k=8
pca=PCA(n_components=k)
X_pca=pca.fit_transform(X_scaled)#(943,8)
'''
#kmeans
cluster=range(2,15)
inertias=[]
for k in cluster:
    kmeans=KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(cluster, inertias, 'bo-')
plt.xlabel('number of clusters')
plt.ylabel('inertias')
plt.title('elbow method for optimal k ')
plt.grid(True)
plt.show()
from sklearn.metrics import silhouette_score
for k in [8,10,12,14]:
     kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
     labels = kmeans.fit_predict(X_pca)
    
     silhouette = silhouette_score(X_pca, labels)
     inertia = kmeans.inertia_
     avg_size = 943 / k
    
     print(f"k={k:2d}: Inertia={inertia:7.1f}, "
          f"Silhouette={silhouette:.3f}, "
          f"平均客户端大小={avg_size:.0f}")
'''
#cluster=10
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X_pca)
df[['user_id', 'cluster']].to_csv('data/user_cluster_mapping.csv', index=False)
print("用户聚类分配已保存到 user_cluster_mapping.csv")

cluster_size=df['cluster'].value_counts().sort_index()
print(cluster_size)

# visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                     c=df['cluster'], cmap='tab10',
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('User Clusters Visualization (2D PCA)', fontsize=14)
plt.colorbar(scatter, label='Client ID')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/cluster_visualization.png', dpi=150)
plt.show()










