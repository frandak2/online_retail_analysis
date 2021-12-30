import seaborn as sns
import matplotlib.pyplot as plt

def boxplot_vis(df):

    plt.figure(figsize=(9, 9))

    plt.subplot(3, 1, 1)
    sns.boxenplot(y='Recency', data=df)
    plt.title('Boxplot of Recency')


    plt.subplot(3, 1, 2)
    sns.boxenplot(y='Frequency', data=df)
    plt.title('Boxplot of Frequency')


    plt.subplot(3, 1, 3)
    sns.boxenplot(y='MonetaryValue', data=df)
    plt.title('Boxplot of Monetary Value')

    plt.tight_layout()