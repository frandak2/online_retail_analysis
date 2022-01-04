import seaborn as sns
import matplotlib.pyplot as plt
from online_retail_analysis.features.build_features import check_skew
import numpy as np


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

def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.1f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)