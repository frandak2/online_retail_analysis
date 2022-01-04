import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Calculation of RFM for Online Retail II dataset
# the dataset is in df_proces
def Calculate_RFMV(df,col):
    max_date=df['InvoiceDate'].max() + pd.Timedelta(days=1)
    df['cost']=df['Quantity']*df['UnitPrice']
    if col == 'StockCode':
        df_rfm = df.groupby([col],as_index=False).agg({'InvoiceDate':lambda x: (max_date-x.max()).days, 
                                                            'cost':'sum',
                                                            'InvoiceNo':'count',
                                                            }).rename(columns = {'InvoiceDate':'Recency',
                                                                                                    'InvoiceNo':'Frequency',
                                                                                                    'cost':'MonetaryValue'})
    else:
        df_rfm = df.groupby([col],as_index=False).agg({'InvoiceDate':lambda x: (max_date-x.max()).days, 
                                                            'cost':'sum',
                                                            'InvoiceNo':'count',
                                                            'StockCode':'nunique',
                                                            }).rename(columns = {'InvoiceDate':'Recency',
                                                                                                    'InvoiceNo':'Frequency',
                                                                                                    'cost':'MonetaryValue',
                                                                                                    'StockCode':'Variety'})
    return df_rfm

def score_rfmv(df):
    rfmv_quantiles = df.iloc[:, 1:].quantile(q = [0.25, 0.5, 0.75]).to_dict()
    def RecencyScore(i, col, df):
        if i <= df[col][0.25]:
            return 4
        elif i <= df[col][0.50]:
            return 3
        elif i <= df[col][0.75]: 
            return 2
        else:
            return 1
        
    # F, M, V: In contrast to Recency, the higher the quantile value, the higher the score    
    def FMVScore(i, col, df):
        if i <= df[col][0.25]:
            return 1
        elif i <= df[col][0.50]:
            return 2
        elif i <= df[col][0.75]: 
            return 3    
        else:
            return 4

    rfmv2 = df.copy()

    rfmv2['R_q'] = rfmv2['Recency'].apply(RecencyScore, args=('Recency', rfmv_quantiles ))
    rfmv2['F_q'] = rfmv2['Frequency'].apply(FMVScore, args=('Frequency', rfmv_quantiles ))
    rfmv2['M_q'] = rfmv2['MonetaryValue'].apply(FMVScore, args=('MonetaryValue', rfmv_quantiles ))
    
    if 'Variety' in rfmv2.columns:
        rfmv2['V_q'] = rfmv2['Variety'].apply(FMVScore, args=('Variety', rfmv_quantiles ))
        rfmv2 = rfmv2[['CustomerID', 'R_q', 'F_q', 'M_q', 'V_q']]
        # # Sum total scores of each component
        df['Total_Score'] = rfmv2['R_q'] + rfmv2['F_q'] + rfmv2['M_q'] + rfmv2['V_q']
        df.index = df['CustomerID']
        df = df.drop('CustomerID', 1)
    else:
        rfmv2 = rfmv2[['StockCode', 'R_q', 'F_q', 'M_q']]
                        # # Sum total scores of each component
        df['Total_Score'] = rfmv2['R_q'] + rfmv2['F_q'] + rfmv2['M_q']
        df.index = df['StockCode']
        df = df.drop('StockCode', 1)
    return df

# def put_labels(segmented_rfm1):
#     label = [0] * len(segmented_rfm1)
#     for i in range(0,len(segmented_rfm1)):
#         if segmented_rfm1['Total_Score'].iloc[i] == '4444':
#             label[i] = "Best Customers"
            
#         elif segmented_rfm1['Total_Score'].iloc[i] == '3344'or segmented_rfm1['Total_Score'].iloc[i] == '4411'or segmented_rfm1['Total_Score'].iloc[i] == '442'or segmented_rfm1['Total_Score'].iloc[i] == '244'or segmented_rfm1['Total_Score'].iloc[i] == '343'or segmented_rfm1['Total_Score'].iloc[i] == '344'or segmented_rfm1['Total_Score'].iloc[i] == '433'or segmented_rfm1['Total_Score'].iloc[i] == '434'or segmented_rfm1['Total_Score'].iloc[i] == '443':
#             label[i] = "Loyal Custumers"
            
#         elif segmented_rfm1['Total_Score'].iloc[i] == '311'or segmented_rfm1['Total_Score'].iloc[i] == '324'or segmented_rfm1['Total_Score'].iloc[i] == '341'or segmented_rfm1['Total_Score'].iloc[i] == '342'or segmented_rfm1['Total_Score'].iloc[i] == '314'or segmented_rfm1['Total_Score'].iloc[i] == '414'or segmented_rfm1['Total_Score'].iloc[i] == '424'or segmented_rfm1['Total_Score'].iloc[i] == '312' or segmented_rfm1['Total_Score'].iloc[i] == '313' or segmented_rfm1['Total_Score'].iloc[i] == '321'or segmented_rfm1['Total_Score'].iloc[i] == '322'or segmented_rfm1['Total_Score'].iloc[i] == '323'or segmented_rfm1['Total_Score'].iloc[i] == '331'or segmented_rfm1['Total_Score'].iloc[i] == '332'or segmented_rfm1['Total_Score'].iloc[i] == '333'or segmented_rfm1['Total_Score'].iloc[i] == '411'or segmented_rfm1['Total_Score'].iloc[i] == '412'or segmented_rfm1['Total_Score'].iloc[i] == '413'or segmented_rfm1['Total_Score'].iloc[i] == '421'or segmented_rfm1['Total_Score'].iloc[i] == '422'or segmented_rfm1['Total_Score'].iloc[i] == '423'or segmented_rfm1['Total_Score'].iloc[i] == '431'or segmented_rfm1['Total_Score'].iloc[i] == '432':
#             label[i] = "Potential Costumers"
#         elif segmented_rfm1['Total_Score'].iloc[i] == '222'or segmented_rfm1['Total_Score'].iloc[i] == '223'or segmented_rfm1['Total_Score'].iloc[i] == '232'or segmented_rfm1['Total_Score'].iloc[i] == '233'or segmented_rfm1['Total_Score'].iloc[i] == '113'or segmented_rfm1['Total_Score'].iloc[i] == '114'or segmented_rfm1['Total_Score'].iloc[i] == '131'or segmented_rfm1['Total_Score'].iloc[i] == '141'or segmented_rfm1['Total_Score'].iloc[i] == '213'or segmented_rfm1['Total_Score'].iloc[i] == '214'or segmented_rfm1['Total_Score'].iloc[i] == '231'or segmented_rfm1['Total_Score'].iloc[i] == '214'or segmented_rfm1['Total_Score'].iloc[i] == '231'or segmented_rfm1['Total_Score'].iloc[i] == '241'or segmented_rfm1['Total_Score'].iloc[i] == '243':
#             label[i] = "Customers Needing Attention"
        
#         elif segmented_rfm1['Total_Score'].iloc[i] == '144'or segmented_rfm1['Total_Score'].iloc[i] == '244'or segmented_rfm1['Total_Score'].iloc[i] == '143'or segmented_rfm1['Total_Score'].iloc[i] == '134':
#             label[i] = "Cant' Lose Them"
#         elif segmented_rfm1['Total_Score'].iloc[i] == '121'or segmented_rfm1['Total_Score'].iloc[i] == '122'or segmented_rfm1['Total_Score'].iloc[i] == '112'or segmented_rfm1['Total_Score'].iloc[i] == '212'or segmented_rfm1['Total_Score'].iloc[i] == '211'or segmented_rfm1['Total_Score'].iloc[i] == '221'or segmented_rfm1['Total_Score'].iloc[i] == '222'or segmented_rfm1['Total_Score'].iloc[i] == '123'or segmented_rfm1['Total_Score'].iloc[i] == '124'or segmented_rfm1['Total_Score'].iloc[i] == '132'or segmented_rfm1['Total_Score'].iloc[i] == '133'or segmented_rfm1['Total_Score'].iloc[i] == '134'or segmented_rfm1['Total_Score'].iloc[i] == '142'or segmented_rfm1['Total_Score'].iloc[i] == '224'or segmented_rfm1['Total_Score'].iloc[i] == '242':
#             label[i] = "At Risk Customers"
#         elif segmented_rfm1['Total_Score'].iloc[i] == '1111':
#             label[i] = "Lost Customers"
            
#         else:
#             label[i] = "Others"
#     segmented_rfm1['label'] = label
#     return segmented_rfm1

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

def rm_outliers(df, col):
    p_05 = df[col].quantile(0.05) # 5th quantile
    p_95 = df[col].quantile(0.95) # 95th quantile
    df[col].clip(p_05, p_95, inplace=True)
    return df