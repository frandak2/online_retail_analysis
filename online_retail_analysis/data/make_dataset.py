import online_retail_analysis.utils.paths as path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(path.data_raw_dir('Online Retail.xlsx')) ## read the dataset
df_proces = df.copy() # create a copy the data
df_proces = df_proces[pd.notnull(df_proces['CustomerID'])] # delete null data from Customer ID
df_proces.drop_duplicates(inplace=True) # remove the duplicated
df_proces['InvoiceNo'] = df_proces['InvoiceNo'].astype('str') #conver to string
df_proces = df_proces[~df_proces['InvoiceNo'].str.contains('C')] # delete Cancel transaccion 
# delete Price and Quantity less than 0
df_proces = df_proces[df_proces.Quantity > 0]
df_proces = df_proces[df_proces.UnitPrice > 0]

for col in ['Quantity','UnitPrice']:
    p_05 = df_proces[col].quantile(0.05) # 5th quantile
    p_95 = df_proces[col].quantile(0.95) # 95th quantile
    df_proces[col].clip(p_05, p_95, inplace=True)

# create graphs
fig, axes= plt.subplots(1,2,figsize=(12,8))
sns.boxplot(y='Quantity', data=df_proces, ax=axes[0])
sns.boxplot(y='UnitPrice', data=df_proces, ax=axes[1])
plt.show()

df_proces.to_csv(path.data_processed_dir('data_clean.csv'),index=False)# export data
