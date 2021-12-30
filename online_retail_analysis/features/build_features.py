import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Calculation of RFM for Online Retail II dataset
# the dataset is in df_proces
def Calculate_RFM(df, labels = False):
    max_date=df['InvoiceDate'].max() + pd.Timedelta(days=1)
    df['cost']=df['Quantity']*df['UnitPrice']
    df_rfm = df.groupby(['CustomerID'],as_index=False).agg({'InvoiceDate':lambda x: (max_date-x.max()).days, 
                                                            'cost':'sum',
                                                            'InvoiceNo':'count'}).rename(columns = {'InvoiceDate':'Recency',
                                                                                                    'InvoiceNo':'Frequency',
                                                                                                    'cost':'MonetaryValue'})
    if labels:

        quartiles = df_rfm.quantile(q=[0.25, 0.5, 0.75])
        def recency_score (data):
            if data <= quartiles.Recency.iloc[0]:
                return 4
            elif data <= quartiles.Recency.iloc[1]:
                return 3
            elif data <= quartiles.Recency.iloc[2]:
                return 2
            else:
                return 1

        def frequency_score (data):
            if data <= quartiles.Frequency.iloc[0]:
                return 1
            elif data <= quartiles.Frequency.iloc[1]:
                return 2
            elif data <= quartiles.Frequency.iloc[2]:
                return 3
            else:
                return 4

        def monetary_value_score (data):
            if data <= quartiles.MonetaryValue.iloc[0]:
                return 1
            elif data <= quartiles.MonetaryValue.iloc[1]:
                return 2
            elif data <= quartiles.MonetaryValue.iloc[2]:
                return 3
            else:
                return 4

        df_rfm['R'] = df_rfm['Recency'].apply(recency_score )
        df_rfm['F'] = df_rfm['Frequency'].apply(frequency_score)
        df_rfm['M'] = df_rfm['MonetaryValue'].apply(monetary_value_score)
        df_rfm['RFM_score'] =df_rfm[['R', 'F', 'M']].sum(axis=1)
        df_rfm['RFM_Segment'] = df_rfm.R.map(str)+df_rfm.F.map(str)+df_rfm.M.map(str)
        segmented_rfm1 = df_rfm
        label = [0] * len(segmented_rfm1)

        for i in range(0,len(segmented_rfm1)):
            if segmented_rfm1['RFM_Segment'].iloc[i] == '444':
                label[i] = "Best Customers"
                
            elif segmented_rfm1['RFM_Segment'].iloc[i] == '334'or segmented_rfm1['RFM_Segment'].iloc[i] == '441'or segmented_rfm1['RFM_Segment'].iloc[i] == '442'or segmented_rfm1['RFM_Segment'].iloc[i] == '244'or segmented_rfm1['RFM_Segment'].iloc[i] == '343'or segmented_rfm1['RFM_Segment'].iloc[i] == '344'or segmented_rfm1['RFM_Segment'].iloc[i] == '433'or segmented_rfm1['RFM_Segment'].iloc[i] == '434'or segmented_rfm1['RFM_Segment'].iloc[i] == '443':
                label[i] = "Loyal Custumers"
                
            elif segmented_rfm1['RFM_Segment'].iloc[i] == '311'or segmented_rfm1['RFM_Segment'].iloc[i] == '324'or segmented_rfm1['RFM_Segment'].iloc[i] == '341'or segmented_rfm1['RFM_Segment'].iloc[i] == '342'or segmented_rfm1['RFM_Segment'].iloc[i] == '314'or segmented_rfm1['RFM_Segment'].iloc[i] == '414'or segmented_rfm1['RFM_Segment'].iloc[i] == '424'or segmented_rfm1['RFM_Segment'].iloc[i] == '312' or segmented_rfm1['RFM_Segment'].iloc[i] == '313' or segmented_rfm1['RFM_Segment'].iloc[i] == '321'or segmented_rfm1['RFM_Segment'].iloc[i] == '322'or segmented_rfm1['RFM_Segment'].iloc[i] == '323'or segmented_rfm1['RFM_Segment'].iloc[i] == '331'or segmented_rfm1['RFM_Segment'].iloc[i] == '332'or segmented_rfm1['RFM_Segment'].iloc[i] == '333'or segmented_rfm1['RFM_Segment'].iloc[i] == '411'or segmented_rfm1['RFM_Segment'].iloc[i] == '412'or segmented_rfm1['RFM_Segment'].iloc[i] == '413'or segmented_rfm1['RFM_Segment'].iloc[i] == '421'or segmented_rfm1['RFM_Segment'].iloc[i] == '422'or segmented_rfm1['RFM_Segment'].iloc[i] == '423'or segmented_rfm1['RFM_Segment'].iloc[i] == '431'or segmented_rfm1['RFM_Segment'].iloc[i] == '432':
                label[i] = "Potential Costumers"

            elif segmented_rfm1['RFM_Segment'].iloc[i] == '222'or segmented_rfm1['RFM_Segment'].iloc[i] == '223'or segmented_rfm1['RFM_Segment'].iloc[i] == '232'or segmented_rfm1['RFM_Segment'].iloc[i] == '233'or segmented_rfm1['RFM_Segment'].iloc[i] == '113'or segmented_rfm1['RFM_Segment'].iloc[i] == '114'or segmented_rfm1['RFM_Segment'].iloc[i] == '131'or segmented_rfm1['RFM_Segment'].iloc[i] == '141'or segmented_rfm1['RFM_Segment'].iloc[i] == '213'or segmented_rfm1['RFM_Segment'].iloc[i] == '214'or segmented_rfm1['RFM_Segment'].iloc[i] == '231'or segmented_rfm1['RFM_Segment'].iloc[i] == '214'or segmented_rfm1['RFM_Segment'].iloc[i] == '231'or segmented_rfm1['RFM_Segment'].iloc[i] == '241'or segmented_rfm1['RFM_Segment'].iloc[i] == '243':
                label[i] = "Customers Needing Attention"
            
            elif segmented_rfm1['RFM_Segment'].iloc[i] == '144'or segmented_rfm1['RFM_Segment'].iloc[i] == '244'or segmented_rfm1['RFM_Segment'].iloc[i] == '143'or segmented_rfm1['RFM_Segment'].iloc[i] == '134':
                label[i] = "Cant' Lose Them"

            elif segmented_rfm1['RFM_Segment'].iloc[i] == '121'or segmented_rfm1['RFM_Segment'].iloc[i] == '122'or segmented_rfm1['RFM_Segment'].iloc[i] == '112'or segmented_rfm1['RFM_Segment'].iloc[i] == '212'or segmented_rfm1['RFM_Segment'].iloc[i] == '211'or segmented_rfm1['RFM_Segment'].iloc[i] == '221'or segmented_rfm1['RFM_Segment'].iloc[i] == '222'or segmented_rfm1['RFM_Segment'].iloc[i] == '123'or segmented_rfm1['RFM_Segment'].iloc[i] == '124'or segmented_rfm1['RFM_Segment'].iloc[i] == '132'or segmented_rfm1['RFM_Segment'].iloc[i] == '133'or segmented_rfm1['RFM_Segment'].iloc[i] == '134'or segmented_rfm1['RFM_Segment'].iloc[i] == '142'or segmented_rfm1['RFM_Segment'].iloc[i] == '224'or segmented_rfm1['RFM_Segment'].iloc[i] == '242':
                label[i] = "At Risk Customers"

            elif segmented_rfm1['RFM_Segment'].iloc[i] == '111':
                label[i] = "Lost Customers"
                
            else:
                label[i] = "Others"

        segmented_rfm1['label'] = label
        return segmented_rfm1
    return df_rfm

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return
