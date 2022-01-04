from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import silhouette_score , davies_bouldin_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import online_retail_analysis.utils.paths as path
from online_retail_analysis.models.funtions import train_elbow_cluster, Kmeans


df_train = pd.read_csv(path.data_processed_dir('data_train_CS.csv')) # read data training
df_score = pd.read_csv(path.data_processed_dir('data_score_CS.csv')) # read data training

X = np.asarray(df_train[['Total_Score']])
train_elbow_cluster(X,finish=150,each=10)
train_elbow_cluster(X,finish=20,each=1)

rfm_k4 = Kmeans(X, 4, df_score.drop('CustomerID',1))
rfm_k5 = Kmeans(X, 5, df_score.drop('CustomerID',1))
rfm_k6 = Kmeans(X, 6, df_score.drop('CustomerID',1))

# lowest better
print('davies score to k=4: {}'.format(davies_bouldin_score(X, rfm_k4.Cluster)))
print('davies score to k=5: {}'.format(davies_bouldin_score(X, rfm_k5.Cluster)))
print('davies score to k=6: {}'.format(davies_bouldin_score(X, rfm_k6.Cluster)))

# higher better
print('davies score to k=4: {}'.format(silhouette_score(X, rfm_k4.Cluster)))
print('davies score to k=5: {}'.format(silhouette_score(X, rfm_k5.Cluster)))
print('davies score to k=6: {}'.format(silhouette_score(X, rfm_k6.Cluster)))

rfm_k6.reset_index().to_csv(path.data_processed_dir('data_clustering.csv'),index=False)
