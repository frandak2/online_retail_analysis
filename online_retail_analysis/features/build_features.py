import online_retail_analysis.utils.paths as path
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer ,StandardScaler
from online_retail_analysis.utils.funtion_features import Calculate_RFMV , score_rfmv , check_skew, rm_outliers

## read data and change the format the date
df_proces = pd.read_csv(path.data_processed_dir('data_clean.csv'))
df_proces['InvoiceDate']= pd.to_datetime(df_proces['InvoiceDate'])

# Data preparation to Costumer segmetation
df_rfm = Calculate_RFMV(df_proces, 'CustomerID')#CustomerID - StockCode
df_rfm_score = score_rfmv(df_rfm)

rfm_copy = df_rfm_score.copy()
rm_outliers(rfm_copy , 'Recency')
rm_outliers(rfm_copy , 'Frequency')
rm_outliers(rfm_copy , 'MonetaryValue')
rm_outliers(rfm_copy , 'Variety')

transformer = FunctionTransformer(np.log)
new_rfm_trans_log = transformer.fit_transform(rfm_copy)
scaler = StandardScaler()
new_rfm_trans_log_sc = scaler.fit_transform(new_rfm_trans_log)
new_rfm_trans_log_sc = pd.DataFrame(new_rfm_trans_log_sc, columns=new_rfm_trans_log.columns)

new_rfm_trans_log_sc.to_csv(path.data_processed_dir('data_train_CS.csv'),index=False)# export data
df_rfm_score.to_csv(path.data_processed_dir('data_score_CS.csv'),index=False)# export data

# Data preparation to Recomender system
df_rfm = Calculate_RFMV(df_proces, 'StockCode')#CustomerID - StockCode
df_rfm_score = score_rfmv(df_rfm)

df = pd.merge(df_rfm_score,df_proces,on="StockCode")
feature_df = df.pivot_table(index='StockCode',columns='CustomerID',values='Total_Score').fillna(0)

feature_df.to_csv(path.data_processed_dir('data_train_RS.csv'),index=False)# export data
df_rfm_score.to_csv(path.data_processed_dir('data_score_RS.csv'),index=False)# export data