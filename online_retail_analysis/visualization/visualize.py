import seaborn as sns
import matplotlib.pyplot as plt
from online_retail_analysis.features.build_features import check_skew


def boxplot_vis(df):

    plt.figure(figsize=(9, 9))

    plt.subplot(4, 1, 1)
    sns.boxplot(y='Recency', data=df)
    plt.title('Boxplot of Recency')


    plt.subplot(4, 1, 2)
    sns.boxplot(y='Frequency', data=df)
    plt.title('Boxplot of Frequency')


    plt.subplot(4, 1, 3)
    sns.boxplot(y='MonetaryValue', data=df)
    plt.title('Boxplot of Monetary Value')

    plt.subplot(4, 1, 4)
    sns.boxplot(y='Variety', data=df)
    plt.title('Boxplot of Variety')
    
    plt.tight_layout()

def plot_distribution_and_skew_test(df):
    plt.figure(figsize=(9, 9))

    plt.subplot(4, 1, 1)
    check_skew(df,'Recency')

    plt.subplot(4, 1, 2)
    check_skew(df,'Frequency')

    plt.subplot(4, 1, 3)
    check_skew(df,'MonetaryValue')

    plt.subplot(4, 1, 4)
    check_skew(df,'Variety')

    plt.tight_layout()