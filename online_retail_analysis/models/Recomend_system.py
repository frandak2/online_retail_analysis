import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import online_retail_analysis.utils.paths as path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


df_train = pd.read_csv(path.data_processed_dir('data_train_RS.csv')) # read data training
df_score = pd.read_csv(path.data_processed_dir('data_score_RS.csv')) # read data training

model_knn = NearestNeighbors(metric ='cosine', algorithm = 'brute')
feature_df_matrix = csr_matrix(df_train.values)
model_knn.fit(feature_df_matrix)
distances, indices = model_knn.kneighbors(df_train.iloc[0,:].values.reshape(1, -1), n_neighbors = 10)

stockCode=[]
cosine=[]
for i in range(0,len(distances.flatten())):
    if i == 0 :
        print("Recommendations for Product : " , df_train.index[0])
    else:
        stockCode.append(df_train.index[indices.flatten()[i]])
        cosine.append(distances.flatten()[i])
        recommendation = pd.DataFrame({"StockCode" : stockCode , "Distance" : cosine })

recommendation=recommendation.sort_values(by='Distance', ascending=False, ignore_index= True)
recommendation


recommendation.reset_index().to_csv(path.data_processed_dir('data_Recomendation.csv'),index=False)
